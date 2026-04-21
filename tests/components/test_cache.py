import torch
import torch.nn.functional as F
from loomlib import SpaceCache, Thunker, ThunkCache

import torch

# Assuming Thunker, SliceCache, and ThunkCache are defined as above

class MyThunker(Thunker):
    def decode(self, x):
        # Just for demo: return mean of values
        return x.float().mean().item()
    def encode(self, value, device=None):
        return torch.tensor(value, device=device)

def print_cache_state(cache, label=""):
    print(f"\n--- {label} ---")
    print("Keys (names):", list(cache.keys()))
    print("All spaces:")
    for name in cache.slices.keys():
        print(f"  {name}: {cache.slices[name].tolist()}")
    print("All thunks:", [repr(t) for t in cache.values()])

def main():
    cache = ThunkCache(device="cpu")

    # 1. Add explicit names
    cache.add("even", torch.tensor([0, 2, 4, 6]), MyThunker())
    cache.add("odd", [1, 3, 5, 7], MyThunker())
    print_cache_state(cache, "After adding 'even' and 'odd' by name")

    # 2. Add via auto-naming (no explicit name)
    cache[torch.tensor([10, 11, 12])] = MyThunker()
    cache[[20, 21]] = MyThunker()
    print_cache_state(cache, "After auto-naming two thunks")

    # 3. Retrieve by name
    print("\nRetrieve by name:")
    print("even:", cache["even"])
    print("odd:", cache["odd"])

    # 4. Retrieve by space
    print("\nRetrieve by space:")
    print(cache[torch.tensor([10, 11, 12])])
    print(cache[[20, 21]])

    # 5. List all names (keys)
    print("\nAll keys:", list(cache.keys()))

    # 6. Call a thunker on its slice
    tensor = torch.arange(30)
    for name, thunk in cache.items():
        indices = cache.slices[name]
        values = tensor.index_select(0, indices)
        print(f"Thunk '{name}' decoded:", thunk.decode(values))

    # 7. Move to (simulated) CUDA if available, then back to CPU
    if torch.cuda.is_available():
        print("\nMoving to cuda:0")
        cache.to("cuda:0")
        print("even's indices (on cuda):", cache.slices["even"].device)
        print("Back to cpu")
        cache.to("cpu")
        print("even's indices (on cpu):", cache.slices["even"].device)

    # 8. Deletion by name and by space
    print("\nDeleting 'even' by name")
    del cache["even"]
    print("Deleting by space (should be auto-named):")
    auto_space = torch.tensor([10, 11, 12])
    del cache[auto_space]
    print_cache_state(cache, "After deletions")

    # 9. Clear everything
    print("\nClearing all entries")
    cache.clear()
    print_cache_state(cache, "After clear()")

if __name__ == "__main__":
    main()

