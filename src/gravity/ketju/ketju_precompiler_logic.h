/* KETJU option logic; included at the end of declarations/precompiler_logic.h */
#ifndef KETJU_PRECOMPILER_LOGIC_H
#define KETJU_PRECOMPILER_LOGIC_H
#if defined(FORCE_EQUAL_TIMESTEPS)
#error "KETJU_REGULARIZATION is not compatible with FORCE_EQUAL_TIMESTEPS"
#endif
#if defined(ADAPTIVE_GRAVSOFT_FORALL) && (ADAPTIVE_GRAVSOFT_FORALL) & 16
#error "KETJU_REGULARIZATION needs fixed star softening: use ADAPTIVE_GRAVSOFT_FORALL without bit 16 and SofteningStars <= KetjuRegionRadius"
#endif
#if defined(GALSF_SINGLE_STARS_FROM_IMF) && defined(GALSF_MERGER_STARCLUSTER_PARTICLES)
#error "GALSF_SINGLE_STARS_FROM_IMF: stars merge only on contact; GALSF_MERGER_STARCLUSTER_PARTICLES merges by distance"
#endif
#if defined(GALSF_SINGLE_STARS_FROM_IMF) && defined(SINGLE_STAR_SINK_DYNAMICS)
#error "GALSF_SINGLE_STARS_FROM_IMF: stars merge only on contact; STARFORGE sink dynamics merges bound sinks within the softening"
#endif
#endif /* KETJU_PRECOMPILER_LOGIC_H */
