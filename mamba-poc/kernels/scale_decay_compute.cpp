// SPDX-License-Identifier: Apache-2.0
// Fused scale + decay compute: scores = scores_raw * D_decay * scale
//
// Uses FPU mul_tiles for scores_raw * D_decay, then SFPU mul_unary_tile for * scale.
// Includes FPU warm-up dummy mul (Blackhole FPU init issue).
//
// Runtime args:
//   0: num_batch (tiles for this core)
//   1: scale (fp32 encoded as uint32)

#include <cstdint>
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"

using std::uint32_t;

void kernel_main() {
    uint32_t num_batch = get_arg_val<uint32_t>(0);
    uint32_t scale_bits = get_arg_val<uint32_t>(1);

    constexpr uint32_t cb_scores = tt::CBIndex::c_0;
    constexpr uint32_t cb_D = tt::CBIndex::c_1;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;

    constexpr uint32_t onetile = 1;

    CircularBuffer scores_cb(cb_scores);
    CircularBuffer D_cb(cb_D);
    CircularBuffer out_cb(cb_out);

    for (uint32_t b = 0; b < num_batch; b++) {
        // Init FPU for mul_tiles
        mul_tiles_init(cb_scores, cb_D);
        // Init SFPU for mul_unary_tile
        binop_with_scalar_tile_init();

        tile_regs_acquire();
        scores_cb.wait_front(onetile);
        D_cb.wait_front(onetile);

        // Dummy mul to reg 0 (FPU warm-up), real mul to reg 1
        mul_tiles(cb_scores, cb_D, 0, 0, 0);  // dummy
        mul_tiles(cb_scores, cb_D, 0, 0, 1);  // real: reg 1 = scores_raw * D

        // Apply scale via SFPU (in-place on reg 1)
        mul_unary_tile(1, scale_bits);

        tile_regs_commit();
        out_cb.reserve_back(onetile);
        tile_regs_wait();
        pack_tile(1, cb_out);
        tile_regs_release();
        scores_cb.pop_front(onetile);
        D_cb.pop_front(onetile);
        out_cb.push_back(onetile);
    }
}
