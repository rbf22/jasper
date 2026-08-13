"""Track malloc stats to find the leak source."""
import gc, ctypes, sys, os, torch, ttnn, struct

sys.path.insert(0, "/home/rfenwick/Documents/jasper/workspace-poc")
from model_ttnn import TTWRAPModel, _safe_deallocate
from train_ttnn import cross_entropy_loss, build_model_config
import yaml

def rss_mb():
    with open("/proc/self/statm") as f:
        return int(f.read().split()[1]) * 4 / 1024

# Use mallinfo2 if available (glibc >= 2.33)
libc = ctypes.CDLL("libc.so.6")

class MallInfo2(ctypes.Structure):
    _fields_ = [
        ("arena", ctypes.c_int),
        ("ordblks", ctypes.c_int),
        ("smblks", ctypes.c_int),
        ("hblks", ctypes.c_int),
        ("hblkhd", ctypes.c_int),
        ("usmblks", ctypes.c_int),
        ("fsmblks", ctypes.c_int),
        ("uordblks", ctypes.c_int),
        ("fordblks", ctypes.c_int),
        ("keepcost", ctypes.c_int),
    ]

try:
    libc.mallinfo2.restype = MallInfo2
    def get_malloc_stats():
        info = libc.mallinfo2()
        return {
            'arena': info.arena,       # total space from sbrk
            'hblkhd': info.hblkhd,     # total space from mmap
            'uordblks': info.uordblks, # total allocated (in use)
            'fordblks': info.fordblks, # total free in arena
        }
    has_mallinfo2 = True
except:
    has_mallinfo2 = False
    def get_malloc_stats():
        return {}

# Also track mmap'd regions count from /proc/self/maps
def mmap_stats():
    try:
        anon_count = 0
        anon_size = 0
        heap_size = 0
        with open("/proc/self/maps") as f:
            for line in f:
                parts = line.split()
                addr_range = parts[0]
                start, end = addr_range.split("-")
                size = int(end, 16) - int(start, 16)
                if "anon" in line or (len(parts) >= 6 and "[anon" in parts[5]):
                    anon_count += 1
                    anon_size += size
                if "[heap]" in line:
                    heap_size = size
        return {'anon_count': anon_count, 'anon_size_kb': anon_size // 1024, 'heap_kb': heap_size // 1024}
    except:
        return {}

device = ttnn.open_device(device_id=0)

with open("configs/cell_a_tt.yaml") as f:
    cfg = yaml.safe_load(f)
cfg["micro_batch_size"] = 0
model_config = build_model_config(cfg)
model = TTWRAPModel(model_config, device)

B, T, V = 8, 128, 128

# Warmup
for _ in range(3):
    input_ids = torch.randint(0, V, (B, T), dtype=torch.int32)
    labels = torch.randint(0, V, (B, T), dtype=torch.int32)
    logits = model.forward(input_ids, k_value=None)
    loss_val, grad_logits = cross_entropy_loss(logits, labels)
    ttnn.synchronize_device(device)
    _safe_deallocate(logits)
    grads = model.backward(grad_logits)
    ttnn.synchronize_device(device)
    _safe_deallocate(grad_logits)
    for g in grads.values():
        _safe_deallocate(g)
    model.clear_caches()

gc.collect()
libc.malloc_trim(0)

rss0 = rss_mb()
mi0 = get_malloc_stats()
mm0 = mmap_stats()
print(f"Baseline: RSS={rss0:.0f}MB")
if has_mallinfo2:
    print(f"  mallinfo2: arena={mi0['arena']//1024}KB hblkhd={mi0['hblkhd']//1024}KB "
          f"uordblks={mi0['uordblks']//1024}KB fordblks={mi0['fordblks']//1024}KB")
print(f"  maps: anon_count={mm0['anon_count']} anon_size={mm0['anon_size_kb']}KB heap={mm0['heap_kb']}KB")

for i in range(200):
    input_ids = torch.randint(0, V, (B, T), dtype=torch.int32)
    labels = torch.randint(0, V, (B, T), dtype=torch.int32)
    logits = model.forward(input_ids, k_value=None)
    loss_val, grad_logits = cross_entropy_loss(logits, labels)
    ttnn.synchronize_device(device)
    _safe_deallocate(logits)
    grads = model.backward(grad_logits)
    ttnn.synchronize_device(device)
    _safe_deallocate(grad_logits)
    for g in grads.values():
        _safe_deallocate(g)
    model.clear_caches()

    if i in {0, 9, 49, 99, 199}:
        gc.collect()
        libc.malloc_trim(0)
        rss = rss_mb()
        mi = get_malloc_stats()
        mm = mmap_stats()
        print(f"  iter {i:3d}: RSS={rss:.0f}MB (+{rss-rss0:.0f})")
        if has_mallinfo2:
            print(f"    mallinfo2: arena={mi['arena']//1024}KB (+{(mi['arena']-mi0['arena'])//1024}) "
                  f"hblkhd={mi['hblkhd']//1024}KB (+{(mi['hblkhd']-mi0['hblkhd'])//1024}) "
                  f"uordblks={mi['uordblks']//1024}KB (+{(mi['uordblks']-mi0['uordblks'])//1024}) "
                  f"fordblks={mi['fordblks']//1024}KB (+{(mi['fordblks']-mi0['fordblks'])//1024})")
        print(f"    maps: anon_count={mm['anon_count']} (+{mm['anon_count']-mm0['anon_count']}) "
              f"anon_size={mm['anon_size_kb']}KB (+{mm['anon_size_kb']-mm0['anon_size_kb']}) "
              f"heap={mm['heap_kb']}KB (+{mm['heap_kb']-mm0['heap_kb']})")

ttnn.close_device(device)
