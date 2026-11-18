# Resources for developing GIZMO with AI coding agents

These files are standing instructions for an AI coding agent (Claude Code, Codex, or similar) working on GIZMO, and a checklist
for the human who supervises it. They condense rules and lessons from a long period of agent-assisted development of this code:
most of them exist because an agent made that mistake, often more than once, and it cost days to find.

The principles themselves are in the User Guide (`docs/gizmo_documentation.md`, section "Code Development & Extensions": code style,
development principles, the loop templates, AI-assisted development). That is the authoritative statement; these files do not
repeat it but turn it into operating rules. They describe the Kokkos (CPU/GPU) development branch and the branches built on it.

| File | Read |
|---|---|
| `rules_for_agents.md` | every session: how to write code here, how to behave, what never to do |
| `workflow_build_test_review_commit.md` | every session that builds, runs, reviews or commits |
| `code_facts_and_pitfalls.md` | before changing or debugging code: facts about GIZMO that agents repeatedly get wrong |

## How to use them

1. Point your agent at them in its own instructions file, kept outside the repository (e.g. `~/.claude/CLAUDE.md` for Claude Code,
   `~/.codex/AGENTS.md` for Codex; an `AGENTS.md` or `CLAUDE.md` inside the checkout is not ignored by git and is easy to commit by
   accident):
   > At the start of every session read, in full, every file in `docs/memory_resources_for_ai_coding/` and the User Guide
   > (`docs/gizmo_documentation.md`) from the heading "Code Development & Extensions" up to "4. Fluid (Hydro) Solvers". Confirm
   > that you have, then wait for the task.

   Agents will claim to have read files they skimmed. If your tool supports hooks (Claude Code does), a hook that refuses every
   other tool until these files have been read in full is worth the five minutes it takes to write.
2. Open each session with your standing principles, then the task. A good opening states the goal, what is in and out of scope,
   and the principles you care about most for this change (single source of truth, no silent descopes, performance at small and
   large active counts, CPU and GPU builds), so the agent applies them to the design rather than discovering them in review.
3. Keep your agent's own memory OUTSIDE the repository: one record document per task (plan, results with their sources, decisions
   and who made them, open items), written as the work happens, not reconstructed at the end. Short session handoffs point to the
   record. None of it is committed: `.gitignore` already excludes `OPEN_*.md`, `STATE_*.md`, `HANDOFF_*.md`, `MEMORY.md`,
   `*_design.md`, `*_handoff.md`, `docs/scratch/`, `.claude/` and `.codex/`.
4. Before designing anything, have the agent search that record and the code for what was already measured or decided; most
   "new" questions have been answered before.
5. Use a second, independent reviewer (another agent or model, given the code rather than the first agent's conclusions) for
   designs and for every non-trivial diff. Where the two disagree, make them argue it out on the evidence; you decide.

## Changing these files

They follow the same rules as the code: terse, general, no references to particular people, machines or private notes. Add a
rule only when it applies to any developer, put it in the file whose topic it belongs to, and check it is not already there.
