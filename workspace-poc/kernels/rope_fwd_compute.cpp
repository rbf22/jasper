// SPDX-License-Identifier: Apache-2.0
// Fused RoPE forward compute: rot_0 = t0*cos - t1*sin, rot_1 = t0*sin + t1*cos
//
// For each (B,T,R,H) batch element:
//   1. p0 = t0 * cos
//   2. p1 = t1 * sin
//   3. rot_0 = p0 - p1
//   4. p2 = t0 * sin
//   5. p3 = t1 * cos
//   6. rot_1 = p2 + p3
//
// Runtime args:
//   0: num_batch (batch elements for this core)

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
    constexpr uint32_t cb_p0 = tt::CBIndex::c_4;
    constexpr uint32_t cb_p1 = tt::CBIndex::c_5;
    constexpr uint32_t cb_p2 = tt::CBIndex::c_6;
    constexpr uint32_t cb_p3 = tt::CBIndex::c_7;
    constexpr uint32_t cb_rot0 = tt::CBIndex::c_16;
    constexpr uint32_t cb_rot1 = tt::CBIndex::c_17;

    constexpr uint32_t onetile = 1;

    CircularBuffer t0_cb(cb_t0);
    CircularBuffer t1_cb(cb_t1);
    CircularBuffer cos_cb(cb_cos);
    CircularBuffer sin_cb(cb_sin);
    CircularBuffer p0_cb(cb_p0);
    CircularBuffer p1_cb(cb_p1);
    CircularBuffer p2_cb(cb_p2);
    CircularBuffer p3_cb(cb_p3);
    CircularBuffer rot0_cb(cb_rot0);
    CircularBuffer rot1_cb(cb_rot1);

    for (uint32_t b = 0; b < num_batch; b++) {
        // Step 1: p0 = t0 * cos
        mul_tiles_init(cb_t0, cb_cos);
        tile_regs_acquire();
        t0_cb.wait_front(onetile);
        cos_cb.wait_front(onetile);
        mul_tiles(cb_t0, cb_cos, 0, 0, 0);
        tile_regs_commit();
        p0_cb.reserve_back(onetile);
        tile_regs_wait();
        pack_tile(0, cb_p0);
        tile_regs_release();
        t0_cb.pop_front(onetile);
        cos_cb.pop_front(onetile);
        p0_cb.push_back(onetile);

        // Step 2: p1 = t1 * sin
        mul_tiles_init(cb_t1, cb_sin);
        tile_regs_acquire();
        t1_cb.wait_front(onetile);
        sin_cb.wait_front(onetile);
        mul_tiles(cb_t1, cb_sin, 0, 0, 0);
        tile_regs_commit();
        p1_cb.reserve_back(onetile);
        tile_regs_wait();
        pack_tile(0, cb_p1);
        tile_regs_release();
        t1_cb.pop_front(onetile);
        sin_cb.pop_front(onetile);
        p1_cb.push_back(onetile);

        // Step 3: rot0 = p0 - p1
        sub_tiles_init(cb_p0, cb_p1);
        tile_regs_acquire();
        p0_cb.wait_front(onetile);
        p1_cb.wait_front(onetile);
        sub_tiles(cb_p0, cb_p1, 0, 0, 0);
        tile_regs_commit();
        rot0_cb.reserve_back(onetile);
        tile_regs_wait();
        pack_tile(0, cb_rot0);
        tile_regs_release();
        p0_cb.pop_front(onetile);
        p1_cb.pop_front(onetile);
        rot0_cb.push_back(onetile);

        // Step 4: p2 = t0 * sin
        mul_tiles_init(cb_t0, cb_sin);
        tile_regs_acquire();
        t0_cb.wait_front(onetile);
        sin_cb.wait_front(onetile);
        mul_tiles(cb_t0, cb_sin, 0, 0, 0);
        tile_regs_commit();
        p2_cb.reserve_back(onetile);
        tile_regs_wait();
        pack_tile(0, cb_p2);
        tile_regs_release();
        t0_cb.pop_front(onetile);
        sin_cb.pop_front(onetile);
        p2_cb.push_back(onetile);

        // Step 5: p3 = t1 * cos
        mul_tiles_init(cb_t1, cb_cos);
        tile_regs_acquire();
        t1_cb.wait_front(onetile);
        cos_cb.wait_front(onetile);
        mul_tiles(cb_t1, cb_cos, 0, 0, 0);
        tile_regs_commit();
        p3_cb.reserve_back(onetile);
        tile_regs_wait();
        pack_tile(0, cb_p3);
        tile_regs_release();
        t1_cb.pop_front(onetile);
        cos_cb.pop_front(onetile);
        p3_cb.push_back(onetile);

        // Step 6: rot1 = p2 + p3
        add_tiles_init(cb_p2, cb_p3);
        tile_regs_acquire();
        p2_cb.wait_front(onetile);
        p3_cb.wait_front(onetile);
        add_tiles(cb_p2, cb_p3, 0, 0, 0);
        tile_regs_commit();
        rot1_cb.reserve_back(onetile);
        tile_regs_wait();
        pack_tile(0, cb_rot1);
        tile_regs_release();
        p2_cb.pop_front(onetile);
        p3_cb.pop_front(onetile);
        rot1_cb.push_back(onetile);
    }
}
