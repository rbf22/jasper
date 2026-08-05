// SPDX-License-Identifier: Apache-2.0
// Fused gate backward reader — reads 3 input tiles per iteration.
// All inputs have the same shape, so tile_id is the same for all.
//
// Runtime args:
//   0: src_gog_addr
//   1: src_gate_addr
//   2: src_of_addr
//   3: num_tiles (tiles for this core)
//   4: tile_start_id
//
// Compile-time args (via TensorAccessorArgs):
//   [0]: gog_args...
//   [gog_offset]: gate_args...
//   [gate_offset]: of_args...

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    uint32_t src_gog_addr = get_arg_val<uint32_t>(0);
    uint32_t src_gate_addr = get_arg_val<uint32_t>(1);
    uint32_t src_of_addr = get_arg_val<uint32_t>(2);
    uint32_t num_tiles = get_arg_val<uint32_t>(3);
    uint32_t tile_start_id = get_arg_val<uint32_t>(4);

    constexpr uint32_t cb_gog  = tt::CBIndex::c_0;
    constexpr uint32_t cb_gate = tt::CBIndex::c_1;
    constexpr uint32_t cb_of   = tt::CBIndex::c_2;

    constexpr auto gog_args = TensorAccessorArgs<0>();
    constexpr auto gate_args = TensorAccessorArgs<gog_args.next_compile_time_args_offset()>();
    constexpr auto of_args = TensorAccessorArgs<gate_args.next_compile_time_args_offset()>();

    const uint32_t gog_tile_bytes = get_tile_size(cb_gog);
    const uint32_t gate_tile_bytes = get_tile_size(cb_gate);
    const uint32_t of_tile_bytes = get_tile_size(cb_of);

    const auto s_gog = TensorAccessor(gog_args, src_gog_addr);
    const auto s_gate = TensorAccessor(gate_args, src_gate_addr);
    const auto s_of = TensorAccessor(of_args, src_of_addr);

    Noc noc;
    CircularBuffer cbgog(cb_gog);
    CircularBuffer cbgate(cb_gate);
    CircularBuffer cbof(cb_of);

    constexpr uint32_t onetile = 1;

    for (uint32_t i = 0; i < num_tiles; i++) {
        uint32_t tile_id = tile_start_id + i;

        cbgog.reserve_back(onetile);
        noc.async_read(s_gog, cbgog, gog_tile_bytes, {.page_id = tile_id}, {.offset_bytes = 0});
        noc.async_read_barrier();
        cbgog.push_back(onetile);

        cbgate.reserve_back(onetile);
        noc.async_read(s_gate, cbgate, gate_tile_bytes, {.page_id = tile_id}, {.offset_bytes = 0});
        noc.async_read_barrier();
        cbgate.push_back(onetile);

        cbof.reserve_back(onetile);
        noc.async_read(s_of, cbof, of_tile_bytes, {.page_id = tile_id}, {.offset_bytes = 0});
        noc.async_read_barrier();
        cbof.push_back(onetile);
    }
}
