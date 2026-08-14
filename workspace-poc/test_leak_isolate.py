#!/usr/bin/env python
"""Isolate which operation leaks by testing each one individually."""
import gc
import os
import sys
import time

import torch
import ttnn

def get_rss_kb():
    with open('/proc/self/status') as f:
        for line in f:
            if line.startswith('VmRSS:'):
                return int(line.split()[1])
    return 0

def test_op(name, fn, device, steps=200):
    """Run fn() steps times and measure RSS growth."""
    # Warmup
    for _ in range(5):
        result = fn()
        if hasattr(result, '__iter__'):
            for r in result:
                try: ttnn.deallocate(r)
                except: pass
        else:
            try: ttnn.deallocate(result)
            except: pass
    ttnn.synchronize_device(device)
    gc.collect()
    
    base = get_rss_kb()
    for step in range(steps):
        result = fn()
        ttnn.synchronize_device(device)
        if hasattr(result, '__iter__'):
            for r in result:
                try: ttnn.deallocate(r)
                except: pass
        else:
            try: ttnn.deallocate(result)
            except: pass
        del result
        if step % 50 == 49:
            gc.collect()
    
    ttnn.synchronize_device(device)
    gc.collect()
    final = get_rss_kb()
    growth = final - base
    rate = growth / steps
    print(f"  {name:>30}: {growth:>8} KB / {steps} steps = {rate:>7.1f} KB/step")
    return rate

def main():
    device = ttnn.open_device(device_id=0)
    
    # Test tensors
    x = ttnn.from_torch(torch.ones([8, 128, 384], dtype=torch.bfloat16), device=device, layout=ttnn.TILE_LAYOUT)
    w = ttnn.from_torch(torch.ones([384, 384], dtype=torch.bfloat16), device=device, layout=ttnn.TILE_LAYOUT)
    x_2d = ttnn.from_torch(torch.ones([8*128, 384], dtype=torch.bfloat16), device=device, layout=ttnn.TILE_LAYOUT)
    scalar = 0.5
    
    print("Isolating leak source (200 steps each):")
    print(f"  Base RSS: {get_rss_kb()} KB")
    print()
    
    # 1. Simple matmul (no scalar)
    test_op("ttnn.matmul", lambda: ttnn.matmul(x_2d, w), device)
    
    # 2. Mul with scalar (binary_ng scalar path)
    test_op("ttnn.mul(x, scalar)", lambda: ttnn.mul(x, scalar), device)
    
    # 3. Add with scalar
    test_op("ttnn.add(x, scalar)", lambda: ttnn.add(x, scalar), device)
    
    # 4. Different scalar each time (tests cache hit)
    scalars = [0.1 * (i % 100) for i in range(1000)]
    idx = [0]
    def vary_scalar():
        s = scalars[idx[0]]
        idx[0] = idx[0] + 1
        return ttnn.mul(x, s)
    test_op("ttnn.mul(x, varying_scalar)", vary_scalar, device)
    
    # 5. ttnn.softmax
    test_op("ttnn.softmax", lambda: ttnn.softmax(x, dim=-1), device)
    
    # 6. ttnn.layer_norm
    weight = ttnn.from_torch(torch.ones([384], dtype=torch.bfloat16), device=device, layout=ttnn.TILE_LAYOUT)
    test_op("ttnn.layer_norm", lambda: ttnn.layer_norm(x, weight=weight, epsilon=1e-6), device)
    
    # 7. ttnn.linear
    test_op("ttnn.linear", lambda: ttnn.linear(x, w), device)
    
    # 8. ttnn.embedding
    ids = torch.randint(0, 128, (8, 128), dtype=torch.long)
    emb_weight = ttnn.from_torch(torch.ones([128, 384], dtype=torch.bfloat16), device=device, layout=ttnn.TILE_LAYOUT)
    test_op("ttnn.embedding", lambda: ttnn.embedding(ids, emb_weight), device)
    
    # 9. ttnn.reshape (view)
    test_op("ttnn.reshape", lambda: ttnn.reshape(x, (8, 128, 384)), device)
    
    # 10. ttnn.transpose
    test_op("ttnn.transpose", lambda: ttnn.transpose(x, -2, -1), device)
    
    ttnn.close_device(device)

if __name__ == "__main__":
    main()
