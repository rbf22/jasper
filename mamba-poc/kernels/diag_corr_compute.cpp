// SPDX-License-Identifier: Apache-2.0
// Fused diagonal correction compute: QK = Q@K^T, QK_g = gamma*QK, out = QK_g@V
//
// For each (B,T,H) batch element:
//   1. QK = Q @ K^T  (1×Nt @ Nt×1 = 1×1 tile, K transposed)
//   2. QK_g = gamma * QK  (scalar broadcast multiply, 1 tile)
//   3. out = QK_g @ V  (1×1 @ 1×Pt = 1×Pt tiles)
//
// Compile-time args:
//   0: Nt (N in tiles, = 2 for N=64)
//   1: Pt (P in tiles, = 2 for P=64)
//
// Runtime args:
//   0: num_batch (batch elements for this core)

#include <cstdint>
#include "api/compute/matmul.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/bcast.h"
#include "api/dataflow/circular_buffer.h"

using std::uint32_t;

void kernel_main() {
    uint32_t num_batch = get_arg_val<uint32_t>(0);
    uint32_t Nt = get_compile_time_arg_val(0);
    uint32_t Pt = get_compile_time_arg_val(1);

    constexpr uint32_t cb_q = tt::CBIndex::c_0;
    constexpr uint32_t cb_k = tt::CBIndex::c_1;
    constexpr uint32_t cb_v = tt::CBIndex::c_2;
    constexpr uint32_t cb_gamma = tt::CBIndex::c_3;
    constexpr uint32_t cb_qk = tt::CBIndex::c_4;       // intermediate: QK
    constexpr uint32_t cb_qk_g = tt::CBIndex::c_5;      // intermediate: QK_g
    constexpr uint32_t cb_out = tt::CBIndex::c_16;       // output

    constexpr uint32_t onetile = 1;

    CircularBuffer q_cb(cb_q);
    CircularBuffer k_cb(cb_k);
    CircularBuffer v_cb(cb_v);
    CircularBuffer gamma_cb(cb_gamma);
    CircularBuffer qk_cb(cb_qk);
    CircularBuffer qk_g_cb(cb_qk_g);
    CircularBuffer out_cb(cb_out);

    for (uint32_t b = 0; b < num_batch; b++) {
        // === 1. QK = Q @ K^T ===
        // Q is (1, Nt) tiles, K is (1, Nt) tiles, K transposed gives (Nt, 1)
        // Output QK is (1, 1) tile
        mm_init(cb_q, cb_k, cb_qk, /*transpose=*/true);

        tile_regs_acquire();
        for (uint32_t kt = 0; kt < Nt; kt++) {
            q_cb.wait_front(onetile);
            k_cb.wait_front(onetile);
            matmul_tiles(cb_q, cb_k, 0, 0, 0);
            q_cb.pop_front(onetile);
            k_cb.pop_front(onetile);
        }
        tile_regs_commit();

        qk_cb.reserve_back(onetile);
        tile_regs_wait();
        pack_tile(0, cb_qk);
        tile_regs_release();
        qk_cb.push_back(onetile);

        // === 2. QK_g = gamma * QK (scalar broadcast) ===
        // gamma is 1 tile (scalar at position 0,0), QK is 1 tile
        // Use mul_tiles_bcast_scalar: multiplies QK by gamma's scalar
        mul_tiles_bcast_scalar_init_short(cb_qk, cb_gamma);
        tile_regs_acquire();
        qk_cb.wait_front(onetile);
        gamma_cb.wait_front(onetile);
        mul_tiles_bcast_scalar(cb_qk, cb_gamma, 0, 0, 0);
        tile_regs_commit();

        qk_g_cb.reserve_back(onetile);
        tile_regs_wait();
        pack_tile(0, cb_qk_g);
        tile_regs_release();
        qk_g_cb.push_back(onetile);

        qk_cb.pop_front(onetile);
        gamma_cb.pop_front(onetile);

        // === 3. out = QK_g @ V ===
        // QK_g is (1, 1) tile, V is (1, Pt) tiles
        // Output is (1, Pt) tiles
        mm_init(cb_qk_g, cb_v, cb_out, /*transpose=*/false);

        for (uint32_t pt = 0; pt < Pt; pt++) {
            tile_regs_acquire();
            qk_g_cb.wait_front(onetile);
            v_cb.wait_front(onetile);
            matmul_tiles(cb_qk_g, cb_v, 0, 0, 0);
            tile_regs_commit();

            out_cb.reserve_back(onetile);
            tile_regs_wait();
            pack_tile(0, cb_out);
            tile_regs_release();
            out_cb.push_back(onetile);

            v_cb.pop_front(onetile);
        }

        qk_g_cb.pop_front(onetile);
    }
}
