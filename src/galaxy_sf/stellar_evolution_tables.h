/* Single-star stellar-evolution tables (stella-tables): installed outside the repository, checked at start-up. */
#ifndef STELLAR_EVOLUTION_TABLES_H
#define STELLAR_EVOLUTION_TABLES_H

#ifdef GALSF_SINGLE_STARS_EVOLUTION_TABLES
void stellar_evolution_tables_init(void);
#include <string>
#include <vector>
int stellar_tables_verify(const std::vector<std::string> &paths); /* stellar_tables_check.cc */
#endif

#endif
