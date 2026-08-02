// SPDX-License-Identifier: Apache-2.0
// Fused RoPE compute — 2 muls per iteration: out0 = t0*cos, out1 = t1*sin
// Uses a dummy mul to a separate dest reg to warm up the FPU.

#include <cstdint>
#include "api/compute/eltwise_binary.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"

using std::uint32_t;

void kernel_main() {
    uint32_t num_batch = get_arg_val<uint32_t>(0);

    constexpr uint32_t cb_t0 = tt::CBIndex::c_0;
    constexpr uint32_t cb_t1 = tt::CBIndex::c_1;
    constexpr uint32_t cb_cos = tt::CBIndex::c_2;
    constexpr uint32_t cb_sin = tt::CBIndex::c_3;
    constexpr uint32_t cb_out0 = tt::CBIndex::c_16;
    constexpr uint32_t cb_out1 = tt::CBIndex::c_17;

    constexpr uint32_t onetile = 1;

    CircularBuffer t0_cb(cb_t0);
    CircularBuffer t1_cb(cb_t1);
    CircularBuffer cos_cb(cb_cos);
    CircularBuffer sin_cb(cb_sin);
    CircularBuffer out0_cb(cb_out0);
    CircularBuffer out1_cb(cb_out1);

    for (uint32_t b = 0; b < num_batch; b++) {
        // out0 = t0 * cos
        mul_tiles_init(cb_t0, cb_cos);
        tile_regs_acquire();
        t0_cb.wait_front(onetile);
        cos_cb.wait_front(onetile);
        // Dummy mul to reg 0 (FPU warm-up), real mul to reg 1
        mul_tiles(cb_t0, cb_cos, 0, 0, 0);  // dummy — discard
        mul_tiles(cb_t0, cb_cos, 0, 0, 1);  // real result in reg 1
        tile_regs_commit();
        out0_cb.reserve_back(onetile);
        tile_regs_wait();
        pack_tile(1, cb_out0);  // pack reg 1 (real result)
        tile_regs_release();
        t0_cb.pop_front(onetile);
        cos_cb.pop_front(onetile);
        out0_cb.push_back(onetile);

        // out1 = t1 * sin
        mul_tiles_init(cb_t1, cb_sin);
        tile_regs_acquire();
        t1_cb.wait_front(onetile);
        sin_cb.wait_front(onetile);
        mul_tiles(cb_t1, cb_sin, 0, 0, 0);
        tile_regs_commit();
        out1_cb.reserve_back(onetile);
        tile_regs_wait();
        pack_tile(0, cb_out1);
        tile_regs_release();
        t1_cb.pop_front(onetile);
        sin_cb.pop_front(onetile);
        out1_cb.push_back(onetile);
    }
}
