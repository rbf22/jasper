// SPDX-License-Identifier: Apache-2.0
// Full fused RoPE compute — single pass:
//   rot1 = x1*cos - x2*sin  (4 FPU muls + 1 SFPU sub)
//   rot2 = x1*sin + x2*cos  (4 FPU muls + 1 SFPU add)
//
// FPU: dummy mul to reg 0 (warm-up), then 4 real muls to regs 1-4.
// SFPU: sub_binary_tile(reg1, reg2, reg0) → rot1
//       add_binary_tile(reg3, reg4, reg1) → rot2

#include <cstdint>
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"

using std::uint32_t;

void kernel_main() {
    uint32_t num_batch = get_arg_val<uint32_t>(0);

    constexpr uint32_t cb_x1   = tt::CBIndex::c_0;
    constexpr uint32_t cb_x2   = tt::CBIndex::c_1;
    constexpr uint32_t cb_cos  = tt::CBIndex::c_2;
    constexpr uint32_t cb_sin  = tt::CBIndex::c_3;
    constexpr uint32_t cb_out0 = tt::CBIndex::c_16;  // rot1
    constexpr uint32_t cb_out1 = tt::CBIndex::c_17;  // rot2

    constexpr uint32_t onetile = 1;

    CircularBuffer x1_cb(cb_x1);
    CircularBuffer x2_cb(cb_x2);
    CircularBuffer cos_cb(cb_cos);
    CircularBuffer sin_cb(cb_sin);
    CircularBuffer out0_cb(cb_out0);
    CircularBuffer out1_cb(cb_out1);

    for (uint32_t b = 0; b < num_batch; b++) {
        // Wait for all 4 input tiles
        x1_cb.wait_front(onetile);
        x2_cb.wait_front(onetile);
        cos_cb.wait_front(onetile);
        sin_cb.wait_front(onetile);

        tile_regs_acquire();

        // Dummy mul to reg 0 (FPU warm-up — first mul is garbage on Blackhole)
        mul_tiles_init(cb_x1, cb_cos);
        mul_tiles(cb_x1, cb_cos, 0, 0, 0);   // reg0 = dummy (discarded)

        // 4 real muls to regs 1-4 (init per pair for safety)
        mul_tiles_init(cb_x1, cb_cos);
        mul_tiles(cb_x1, cb_cos, 0, 0, 1);   // reg1 = x1 * cos
        mul_tiles_init(cb_x2, cb_sin);
        mul_tiles(cb_x2, cb_sin, 0, 0, 2);   // reg2 = x2 * sin
        mul_tiles_init(cb_x1, cb_sin);
        mul_tiles(cb_x1, cb_sin, 0, 0, 3);   // reg3 = x1 * sin
        mul_tiles_init(cb_x2, cb_cos);
        mul_tiles(cb_x2, cb_cos, 0, 0, 4);   // reg4 = x2 * cos

        // SFPU binary ops on dest regs
        sub_binary_tile_init();
        sub_binary_tile(1, 2, 0);             // reg0 = reg1 - reg2 = x1*cos - x2*sin = rot1
        add_binary_tile_init();
        add_binary_tile(3, 4, 1);             // reg1 = reg3 + reg4 = x1*sin + x2*cos = rot2

        tile_regs_commit();

        out0_cb.reserve_back(onetile);
        out1_cb.reserve_back(onetile);
        tile_regs_wait();
        pack_tile(0, cb_out0);                // pack rot1
        pack_tile(1, cb_out1);                // pack rot2
        tile_regs_release();

        x1_cb.pop_front(onetile);
        x2_cb.pop_front(onetile);
        cos_cb.pop_front(onetile);
        sin_cb.pop_front(onetile);
        out0_cb.push_back(onetile);
        out1_cb.push_back(onetile);
    }
}
