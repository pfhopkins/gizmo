"""Giant-impact bodies for GIZMO from one place: the equation-of-state tables (verified by sha256), the body presets (proto ice giants,
impactors, the Sandnes et al. 2025 Jupiter validation pair), the composition-bin mixture tables of graded interiors, the particle IC and
the matching params block. Every body of a production campaign is built here, so it is reproducible from the recorded table hashes and
arguments (each IC gets a .manifest.json beside it).

TABLES are read from a directory given with --tables (normally the IC library's eos_tables/, where each table carries a .json sidecar with
its construction, inputs and licence); a table whose sha256 differs from the one recorded below is refused.

MATERIAL SLOTS (the AneosTable index), the same for every body so that a target and an impactor combine into one run: 0 H/He (CD21,
Y 0.275; Y 0.245 for the Sandnes pair), 1 water, 2 forsterite (Stewart 2019), 3 iron (Stewart 2020), 4 .. 4+K-1 the K graded-zone bins
(outermost, most water-rich, first). The CompositionType a cell carries depends on the build (--layout, REQUIRED, no default; read the
build's effective settings, printed at the top of every run, since EOS_TILLOTSON is also switched on by EOS_DAMAGE_POROSITY, PLANET_HEATING
and GRAIN_FLUID_PROMOTION): 'aneos' (EOS_ANEOS without EOS_TILLOTSON): CompositionType = slot; 'tillotson+aneos' (EOS_TILLOTSON and
EOS_ANEOS, with or without EOS_TYPES_DEFAULTGAS_AND_SOLIDS): CompositionType = 7 + slot, and the params block then also carries the
(unused) custom Tillotson material, whose parameters such a build requires; 'defaultgas+aneos' (EOS_ANEOS and EOS_TYPES_DEFAULTGAS_AND_SOLIDS
without EOS_TILLOTSON, where CompositionType 0 is the ordinary gas): CompositionType = 1 + slot. In the wrong layout, cells would run as
other materials.

BODIES are hydrostatic, on the first-law adiabat (T continuous across every boundary). A body with an H/He envelope is truncated at the
pressure P_t (no particles above it); its envelope enters there at the temperature of the ideal H2 + He adiabat through the 1-bar
temperature T_1bar (hse_solver.ideal_gas_atmosphere). A condensed body (no envelope) is truncated at P_s = 1 bar with surface temperature T_s.
A graded zone (water above -> forsterite below, rock mass fraction rising linearly in mass from 0 to 1) is represented by K equal-mass bins of
mean rock fraction (k + 1/2) / K, each a mixture table (mix_tables.py); the pure water layer above and the bins are placed as one set of
shells (placement group), composition assigned by enclosed mass.

    python3 -m initial_conditions.eos_tools.giant_impact_bodies list
    python3 -m initial_conditions.eos_tools.giant_impact_bodies bins <K> --tables DIR --bins DIR
    python3 -m initial_conditions.eos_tools.giant_impact_bodies body <preset> <N> <out.hdf5> --tables DIR --layout LAYOUT --kernel KERNEL_FUNCTION
            --ngb DesNumNgb [--K K --bins DIR] [--T T_1bar_or_T_s] [--mass M_E] [--id-offset n] [--calibrate particle|shell|none]
    python3 -m initial_conditions.eos_tools.giant_impact_bodies params <out.params_block> <preset> [<preset> ...] --tables DIR --layout LAYOUT
            [--K K --bins DIR --bin-material strengthless|<preset number>]
"""
import argparse, hashlib, json, os, subprocess, sys
import numpy as np

M_EARTH = 5.9722e27            # g
P_ENVELOPE_TRUNCATION = 1e8    # dyn/cm^2 (0.01 GPa): inside the CD21 table on every planned adiabat, far below the first particle layer
P_CONDENSED_SURFACE = 1e6      # dyn/cm^2 (1 bar)

# file name in the table directory, sha256, Helium mass fraction (H/He only), the heat capacity the IC builder uses for temperature-mode
# inputs (erg/g/K; it does not enter a tabulated material's states)
TABLES = {
    "hhe":       ("cd21_y0275.sesame",        "4d971be9653bcb4ff23c140746d2ea4139b49d468eff4b8977f5f10911aae62a", 0.275, 1e8),
    "hhe_y0245": ("cd21_y0245.sesame",        "58ac77e5fcc6c532431186f8b00c50931c73bb354a5a1df70aa06e955bb88075", 0.245, 1e8),
    "water":     ("water_baseline.sesame",    "13d37e09dbefb620b8496450862dec17eb904e3ee5603cdaf91e7facbfe259ab", None, 4e7),
    "forsterite": ("forsterite_S19_T4.sesame", "b5921224c54dbab80de721509bbe3e7e8f192694027e422ce35f4aa41913521f", None, 1e7),
    "iron":      ("iron_S20_T4.sesame",       "36a17ca3a9ddcadea19ae7b4e7348c66ac2dcce09dc4129380ce2662b223f610", None, 4.5e6),
}
SLOT = {"hhe": 0, "hhe_y0245": 0, "water": 1, "forsterite": 2, "iron": 3}
FIRST_BIN_SLOT = 4
N_TILLOTSON_MATERIALS = 7      # src/eos/composition_registry.h: ANEOS tables follow the Tillotson materials in a build with both
LAYOUT_OFFSET = {"aneos": 0, "tillotson+aneos": N_TILLOTSON_MATERIALS, "defaultgas+aneos": 1}

# solid material of each table in the params block (EOS_ELASTIC builds): a preset number, or None = strengthless (own zero values)
SOLID_MATERIAL = {"hhe": None, "hhe_y0245": None, "water": None, "forsterite": 5, "iron": 3}   # presets: 5 olivine/dunite, 3 iron

# presets: total mass (M_E), layers outer -> inner as (material, mass in M_E) with ("graded", mass) for a water -> forsterite zone,
# envelope H/He table (or None for a condensed body), default T_1bar (envelope) or T_s (condensed) [K]
PRESETS = {
    "uranus":          dict(layers=[("hhe", 2.0), ("water", 9.0), ("forsterite", 1.1), ("iron", 0.4)], envelope="hhe", T=100.0),
    "uranus_graded":   dict(layers=[("hhe", 2.0), ("water", 7.9), ("graded", 2.2), ("iron", 0.4)], envelope="hhe", T=100.0),
    "neptune":         dict(layers=[("hhe", 2.5), ("water", 10.3), ("forsterite", 1.7), ("iron", 0.7)], envelope="hhe", T=100.0),
    "impactor_differentiated": dict(layers=[("water", 1.0), ("forsterite", 0.67), ("iron", 0.33)], envelope=None, T=200.0),
    "impactor_water":  dict(layers=[("water", 2.0)], envelope=None, T=200.0),
    "impactor_rocky":  dict(layers=[("forsterite", 4.0 / 3.0), ("iron", 2.0 / 3.0)], envelope=None, T=1500.0),
    "sandnes_jupiter": dict(layers=[("hhe_y0245", 298.0), ("water", 10.0)], envelope="hhe_y0245", T=165.0),
    "sandnes_impactor": dict(layers=[("water", 10.0)], envelope=None, T=200.0),
}


def sha256_file(path, block=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(block), b""): h.update(b)
    return h.hexdigest()


def table_path(name, table_dir):
    """the verified path of a component table: refuses a missing file or a hash that differs from the recorded one"""
    fn, sha, _, _ = TABLES[name]
    path = os.path.join(table_dir, fn)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{path}: table '{name}' not found in {table_dir}")
    got = sha256_file(path)
    if got != sha:
        raise ValueError(f"{path}: sha256 {got} differs from the recorded {sha} (table '{name}'): refused")
    return path


def bin_fractions(K):
    """rock (forsterite) mass fraction of each graded bin, outermost first"""
    return [(k + 0.5) / K for k in range(K)]


def bin_path(K, k, bin_dir):
    return os.path.join(bin_dir, f"graded_water_forsterite_K{K}_k{k:02d}.sesame")


def write_bins(K, table_dir, bin_dir, accept_u_order_pairs=False):
    """the K mixture tables of a graded water -> forsterite zone (mix_tables.make on the verified component tables); returns their paths
    and sha256 (also written to bin_dir/graded_K<K>.json). accept_u_order_pairs: see mix_tables.make (water_baseline's accepted cold corner,
    ~600 GPa near 250 K, passes into every bin)"""
    from .mix_tables import make
    w, fo = table_path("water", table_dir), table_path("forsterite", table_dir)
    os.makedirs(bin_dir, exist_ok=True)
    out = []
    for k, x in enumerate(bin_fractions(K)):
        p = bin_path(K, k, bin_dir)
        make(p, [(w, 1.0 - x), (fo, x)], accept_u_order_pairs=accept_u_order_pairs)
        out.append(dict(file=os.path.basename(p), rock_fraction=x, sha256=sha256_file(p)))
    manifest = dict(K=K, components={"water": TABLES["water"][1], "forsterite": TABLES["forsterite"][1]}, accept_u_order_pairs=accept_u_order_pairs, bins=out)
    text = json.dumps(manifest, indent=1) + "\n"
    with open(os.path.join(bin_dir, f"graded_K{K}.json"), "w") as f: f.write(text)
    return out


def verified_bins(K, bin_dir, table_dir):
    """paths of the K bin tables, checked against the manifest write_bins left beside them (and that manifest against today's
    component hashes)"""
    mpath = os.path.join(bin_dir, f"graded_K{K}.json")
    if not os.path.isfile(mpath):
        raise FileNotFoundError(f"{mpath}: build the bins first ('bins {K}')")
    man = json.load(open(mpath))
    if man["components"] != {"water": TABLES["water"][1], "forsterite": TABLES["forsterite"][1]}:
        raise ValueError(f"{mpath}: built from other component tables than the recorded ones: rebuild the bins")
    want = bin_fractions(K)
    if (man.get("K") != K or len(man["bins"]) != K
            or [b["file"] for b in man["bins"]] != [os.path.basename(bin_path(K, k, bin_dir)) for k in range(K)]
            or any(abs(b["rock_fraction"] - x) > 1e-12 for b, x in zip(man["bins"], want))):
        raise ValueError(f"{mpath}: does not describe K = {K} bins in order with rock fractions (k + 1/2) / K: rebuild the bins")
    paths = []
    for k, b in enumerate(man["bins"]):
        p = os.path.join(bin_dir, b["file"])
        if sha256_file(p) != b["sha256"]:
            raise ValueError(f"{p}: sha256 differs from its manifest: refused")
        paths.append(p)
    return paths


def builder_provenance():
    """the GIZMO commit, the builder sources (this directory's .py files) that differ from it, and every source's sha256"""
    here = os.path.dirname(os.path.abspath(__file__))
    def git(*args):
        try:
            return subprocess.run(["git", "-C", here, *args], capture_output=True, text=True).stdout
        except OSError:
            return ""
    commit = git("rev-parse", "HEAD").strip()
    status = git("status", "--porcelain", "--untracked-files=all", "--", ".")     # lines "XY path": never strip the leading status column
    modified = sorted(line[3:].strip() for line in status.splitlines() if line[3:].strip().endswith(".py"))
    sources = {fn: sha256_file(os.path.join(here, fn)) for fn in sorted(os.listdir(here)) if fn.endswith(".py")}
    return dict(commit=commit, modified=modified, sources_sha256=sources)


def zones_for(preset, table_dir, layout, mass=None, K=0, bin_dir=None):
    """hse_solver zones (outer -> inner) of a preset, its total mass [g] and the list of (slot, table path) it uses; each zone's
    CompositionType = the build layout's offset + its slot"""
    offset = LAYOUT_OFFSET[layout]
    from .eos_dispatch import Material
    from .hse_solver import Zone
    spec = PRESETS[preset]
    layers = list(spec["layers"])
    M = sum(m for _, m in layers)
    scale = (mass / M) if mass else 1.0
    used, mats = {}, []
    for name, m in layers:
        if name == "graded":
            if K < 1 or not bin_dir:
                raise ValueError(f"preset {preset} has a graded zone: give --K and --bins")
            for k, (p, x) in enumerate(zip(verified_bins(K, bin_dir, table_dir), bin_fractions(K))):
                slot = FIRST_BIN_SLOT + k
                mats.append((Material.sesame(f"bin x={x:.4f}", p, offset + slot, 2e7), m * scale / K, "graded")); used[slot] = p
        else:
            p = table_path(name, table_dir)
            mats.append((Material.sesame(name, p, offset + SLOT[name], TABLES[name][3]), m * scale, "graded" if (name == "water" and any(n == "graded" for n, _ in layers)) else None))
            used[SLOT[name]] = p
    zones, below = [], M * scale
    for i, (mat, m, group) in enumerate(mats):
        below -= m
        frac = 0.0 if i == len(mats) - 1 else below / (M * scale)
        zones.append(Zone(mat, frac, placement_group=group) if group else Zone(mat, frac))
    return zones, M * scale * M_EARTH, used


def build_body(preset, N, out, table_dir, layout, kernel, ngb, T=None, mass=None, K=0, bin_dir=None, id_offset=0, calibrate="particle"):
    from .build_layered_body import build_planet
    from .hse_solver import ideal_gas_atmosphere
    spec = PRESETS[preset]
    zones, M, used = zones_for(preset, table_dir, layout, mass, K, bin_dir)
    T = spec["T"] if T is None else T
    if spec["envelope"]:
        atm = ideal_gas_atmosphere(T, TABLES[spec["envelope"]][2], P_ENVELOPE_TRUNCATION, M)
        T_entry, P_entry = atm["T_t"], P_ENVELOPE_TRUNCATION
    else:
        T_entry, P_entry = T, P_CONDENSED_SURFACE
    placed, prof = build_planet(zones, M, T_entry, P_entry, N, out, kernel=kernel, des_num_ngb=ngb, id_offset=id_offset,
                                calibrate_masses=False if calibrate == "none" else calibrate)
    builder = builder_provenance()
    if builder["modified"]:
        print(f"WARNING: the IC builder sources differ from commit {builder['commit'] or '(unknown)'}: {', '.join(builder['modified'])} "
              "(recorded in the manifest with every source's sha256)")
    manifest = dict(preset=preset, layers=PRESETS[preset]["layers"], mass_M_E=M / M_EARTH, N_requested=N, N_placed=int(len(placed["coords"])),
                    layout=layout, composition_type_offset=LAYOUT_OFFSET[layout], T=T, T_entry=T_entry, P_entry=P_entry, kernel=kernel, DesNumNgb=ngb, K=K, id_offset=id_offset, calibrate=calibrate,
                    radius_cm=float(prof["R"]), tables={str(s): dict(file=os.path.basename(p), sha256=sha256_file(p)) for s, p in sorted(used.items())},
                    builder=builder)
    text = json.dumps(manifest, indent=1) + "\n"
    with open(out + ".manifest.json", "w") as f: f.write(text)
    print(f"wrote {out} ({manifest['N_placed']} cells, R {prof['R']:.5e} cm) + {out}.manifest.json")
    return placed, prof, manifest


def params_block(table_dir, presets, layout, K=0, bin_dir=None, bin_material=None):
    """the AneosNumMaterials / AneosTable<N> / AneosTable<N>_SolidMaterial lines for a run of the bodies `presets` (e.g. a target and its
    impactor): the slots they use, at their fixed numbers; a lower slot no body uses gets the (small) H/He table, so the numbering stays
    fixed without loading unused large tables. bin_material: how graded bins carry strength in EOS_ELASTIC builds -- REQUIRED when a body
    has a graded zone (no default): 'strengthless', or a preset number (1-6)."""
    envelopes = {PRESETS[p]["envelope"] for p in presets if PRESETS[p]["envelope"]}
    if len(envelopes) > 1:
        raise ValueError(f"bodies with different H/He tables ({sorted(envelopes)}) cannot share slot 0")
    hhe = envelopes.pop() if envelopes else "hhe"
    names = {SLOT[n] if n in SLOT else None: n for p in presets for n, _ in PRESETS[p]["layers"] if n != "graded"}
    graded = any(n == "graded" for p in presets for n, _ in PRESETS[p]["layers"])
    paths, kind = {}, {}
    for s, n in names.items(): paths[s], kind[s] = table_path(n, table_dir), n
    if graded:
        if K < 1 or not bin_dir: raise ValueError("a body has a graded zone: give --K and --bins")
        if bin_material is None:
            raise ValueError("graded bins: give --bin-material ('strengthless' or a preset number); there is no default")
        if bin_material != "strengthless" and bin_material not in [str(n) for n in range(1, N_TILLOTSON_MATERIALS)]:
            raise ValueError(f"--bin-material {bin_material}: 'strengthless' or a preset number 1-{N_TILLOTSON_MATERIALS - 1}")
        for k, p in enumerate(verified_bins(K, bin_dir, table_dir)): paths[FIRST_BIN_SLOT + k], kind[FIRST_BIN_SLOT + k] = p, "bin"
    for s in range(max(paths)):
        if s not in paths: paths[s], kind[s] = table_path(hhe, table_dir), "unused"
    lines = [f"AneosNumMaterials                  {len(paths)}"]
    for s in sorted(paths):
        note = "  (slot unused by these bodies)" if kind[s] == "unused" else ""
        lines.append(f"AneosTable{s:<2d}                       {paths[s]}      % sha256 {sha256_file(paths[s])[:16]}{note}")
    lines.append("% with EOS_ELASTIC every table names its solid material: a preset (5 olivine, 3 iron) or 0 = its own values (here zero strength)")
    for s in sorted(paths):
        if kind[s] == "bin":
            mat = None if bin_material == "strengthless" else int(bin_material)
        elif kind[s] == "unused":
            mat = None
        else:
            mat = SOLID_MATERIAL[kind[s]]
        if mat is not None:
            lines.append(f"AneosTable{s}_SolidMaterial          {mat}")
        else:
            lines += [f"AneosTable{s}_SolidMaterial          0      % strengthless (fluid)",
                      f"AneosTable{s}_ShearModulus           0", f"AneosTable{s}_YieldStrength          0",
                      f"AneosTable{s}_WeibullK               0      % no flaws: never damaged", f"AneosTable{s}_WeibullM               9",
                      f"AneosTable{s}_InternalFriction       0", f"AneosTable{s}_Cohesion               0",
                      f"AneosTable{s}_YieldStrengthMax       0", f"AneosTable{s}_FractureActivation     0",
                      f"AneosTable{s}_ReferenceDensity       1.0    % only sets the (unused) flaw volume"]
    if layout == "tillotson+aneos":
        lines += [f"% the tables are CompositionType {N_TILLOTSON_MATERIALS} + slot in this build; the custom Tillotson material (CompositionType 0) is not used",
                  "% by these bodies, but a build with EOS_TILLOTSON reads its parameters at startup: the user-guide iron example values",
                  "Tillotson_EOS_params_a             0.5", "Tillotson_EOS_params_b             1.5", "Tillotson_EOS_params_u_0           9.5e10",
                  "Tillotson_EOS_params_rho_0         7.86", "Tillotson_EOS_params_A             1.28e12", "Tillotson_EOS_params_B             1.05e12",
                  "Tillotson_EOS_params_u_s           1.42e10", "Tillotson_EOS_params_u_s_prime     8.45e10", "Tillotson_EOS_params_alpha         5.0",
                  "Tillotson_EOS_params_beta          5.0"]
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["list", "bins", "body", "params"])
    ap.add_argument("args", nargs="*")
    ap.add_argument("--tables"); ap.add_argument("--bins"); ap.add_argument("--K", type=int, default=0)
    ap.add_argument("--layout", choices=sorted(LAYOUT_OFFSET), help="the build's equation-of-state layout (required for body / params)")
    ap.add_argument("--accept-u-order-pairs", action="store_true", help="bins: accept u not rising with T inherited from a component")
    ap.add_argument("--kernel", type=int); ap.add_argument("--ngb", type=float)
    ap.add_argument("--T", type=float); ap.add_argument("--mass", type=float); ap.add_argument("--id-offset", type=int, default=0)
    ap.add_argument("--calibrate", choices=["particle", "shell", "none"], default="particle")
    ap.add_argument("--bin-material")
    a = ap.parse_args(argv)
    if a.mode == "list":
        for name, p in PRESETS.items():
            print(f"{name:24s} {sum(m for _, m in p['layers']):7.2f} M_E  {p['layers']}  {'T_1bar' if p['envelope'] else 'T_s'} {p['T']:g} K")
        return
    if not a.tables:
        ap.error("--tables DIR is required")
    for p in (a.args[:1] if a.mode == "body" else a.args[1:] if a.mode == "params" else []):
        if p not in PRESETS: ap.error(f"unknown preset '{p}' (list shows them)")
    if a.mode == "bins":
        if len(a.args) != 1 or not a.bins: ap.error("bins <K> --bins DIR")
        write_bins(int(a.args[0]), a.tables, a.bins, a.accept_u_order_pairs)
    elif a.mode == "body":
        if len(a.args) != 3 or a.kernel is None or a.ngb is None or a.layout is None:
            ap.error("body <preset> <N> <out.hdf5> --layout LAYOUT --kernel K --ngb N")
        build_body(a.args[0], int(float(a.args[1])), a.args[2], a.tables, a.layout, a.kernel, a.ngb, a.T, a.mass, a.K, a.bins, a.id_offset, a.calibrate)
    elif a.mode == "params":
        if len(a.args) < 2 or a.layout is None: ap.error("params <out.params_block> <preset> [<preset> ...] --layout LAYOUT")
        text = params_block(a.tables, a.args[1:], a.layout, a.K, a.bins, a.bin_material)
        with open(a.args[0], "w") as f: f.write(text)
        print(text, end="")


if __name__ == "__main__":
    main()
