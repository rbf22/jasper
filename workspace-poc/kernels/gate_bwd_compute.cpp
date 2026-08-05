// SPDX-License-Identifier: Apache-2.0
// Fused gate backward compute — single pass:
//   out0 = grad_out_gated * gate                    (grad_out_flat)
//   out1 = grad_out_gated * out_flat * gate * (1-g) (grad_g)
//
// FPU: dummy mul (warm-up) + 3 real muls to dest regs 1-3
// SFPU: copy gate to reg, then mul/sub/mul chain for grad_g

#include <cstdint>
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"

using std::uint32_t;

void kernel_main() {
    uint32_t num_batch = get_arg_val<uint32_t>(0);

    constexpr uint32_t cb_gog  = tt::CBIndex::c_0;  // grad_out_gated
    constexpr uint32_t cb_gate = tt::CBIndex::c_1;  // gate (sigmoid(g))
    constexpr uint32_t cb_of   = tt::CBIndex::c_2;  // out_flat
    constexpr uint32_t cb_out0 = tt::CBIndex::c_16; // grad_out_flat
    constexpr uint32_t cb_out1 = tt::CBIndex::c_17; // grad_g

    constexpr uint32_t onetile = 1;

    CircularBuffer gog_cb(cb_gog);
    CircularBuffer gate_cb(cb_gate);
    CircularBuffer of_cb(cb_of);
    CircularBuffer out0_cb(cb_out0);
    CircularBuffer out1_cb(cb_out1);

    for (uint32_t b = 0; b < num_batch; b++) {
        gog_cb.wait_front(onetile);
        gate_cb.wait_front(onetile);
        of_cb.wait_front(onetile);

        tile_regs_acquire();

        // Dummy mul to reg 0 (FPU warm-up — first mul is garbage on Blackhole)
        mul_tiles_init(cb_gog, cb_gate);
        mul_tiles(cb_gog, cb_gate, 0, 0, 0);   // reg0 = dummy (discarded)

        // 3 real FPU muls
        mul_tiles_init(cb_gog, cb_gate);
        mul_tiles(cb_gog, cb_gate, 0, 0, 1);   // reg1 = gog * gate = grad_out_flat
        mul_tiles_init(cb_gog, cb_of);
        mul_tiles(cb_gog, cb_of, 0, 0, 2);     // reg2 = gog * out_flat
        mul_tiles_init(cb_gate, cb_gate);
        mul_tiles(cb_gate, cb_gate, 0, 0, 3);  // reg3 = gate * gate = gate^2

        // SFPU chain for grad_g = (gog * out_flat) * (gate - gate^2)
        copy_tile_init(cb_gate);
        copy_tile(cb_gate, 0, 4);              // reg4 = gate (copy to dest for SFPU)
        sub_binary_tile_init();
        sub_binary_tile(4, 3, 5);              // reg5 = gate - gate^2 = sig_prime
        mul_binary_tile_init();
        mul_binary_tile(2, 5, 6);              // reg6 = (gog*out_flat) * sig_prime = grad_g

        tile_regs_commit();

        out0_cb.reserve_back(onetile);
        out1_cb.reserve_back(onetile);
        tile_regs_wait();
        pack_tile(1, cb_out0);                 // pack grad_out_flat
        pack_tile(6, cb_out1);                 // pack grad_g
        tile_regs_release();

        gog_cb.pop_front(onetile);
        gate_cb.pop_front(onetile);
        of_cb.pop_front(onetile);
        out0_cb.push_back(onetile);
        out1_cb.push_back(onetile);
    }
}
