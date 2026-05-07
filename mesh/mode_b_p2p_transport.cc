/* mesh/mode_b_p2p_transport.cc
 *
 * Out-of-line bits for the templated P2P transport: the global diagnostic
 * counter and its reset/snapshot helpers. Templates themselves are in the
 * header.
 */

#include "mode_b_p2p_transport.h"
#include <cstring>

mode_b_p2p_diag_t g_mode_b_p2p_diag = {0, 0, 0, 0, 0, 0};

void mode_b_p2p_diag_reset(void)
{
    std::memset(&g_mode_b_p2p_diag, 0, sizeof(g_mode_b_p2p_diag));
}

mode_b_p2p_diag_t mode_b_p2p_diag_snapshot(void)
{
    return g_mode_b_p2p_diag;
}
