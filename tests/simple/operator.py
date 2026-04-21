from loomlib import ActionSpace, FnSpace
from loomlib.thunker import OneHotIntDecoder, VocabDecoder
from tensor_mosaic import Mosaic
from dataclasses import dataclass
import torch

# Helper to get function argument names
@dataclass
class DE:
    e1: int
    e2: int
    o1: int
    o2: int
    a: float
    action: str = "DE"

@dataclass
class SE:
    e1: int
    o1: int
    a: float
    action: str = "SE"

# --- Usage Example ---
# Suppose you have a set of thunker decoders, e.g. OneHotIntDecoder, VocabDecoder, etc.
actions = ["SE", "DE"]
angle_vals = [2**(-i) / 320 for i in range(-5, 5)]
actionThunker = VocabDecoder(actions)
angleThunker = VocabDecoder(angle_vals)
q = {}
num_electrons = 8
num_qubits = 16

q[0] = VocabDecoder([i for i in range(num_electrons)])
q[1] = VocabDecoder([i for i in range(num_electrons)])
q[2] = VocabDecoder([i for i in range(num_electrons, num_qubits)])
q[3] = VocabDecoder([i for i in range(num_electrons, num_qubits)])

mosaic = Mosaic(dim=1)

# Allocate space for each variable
mosaic.action = 2
mosaic.angle_vals = len(angle_vals)

mosaic.e1 = len(q[0])
mosaic.e2 = len(q[1])
mosaic.o1 = len(q[2])
mosaic.o2 = len(q[3])




fnspace = FnSpace(dim=1)
fnspace.add_type(
    action=actionThunker,
    e1=q[0], e2=q[1], o1=q[2], o2=q[3],
    a=angleThunker,
)

def decode_fn(action, e1, e2, o1, o2, a):
    # Dummy
    return list(zip(action, e1, e2, o1, o2, a))

fnspace.add_fn("as_struct", decode_fn)

x = torch.randn(4, *fnspace.mosaic.shape)
decoded = fnspace(x, return_types=True)
print(decoded)

batch = [
    SE(e1=1, o1=9, a=angle_vals[0]),
    DE(e1=3, o1=10, e2=5, o2=13, a=angle_vals[1]),
    SE(e1=7, o1=12, a=angle_vals[2]),
]

logits = fnspace.encode(batch)



print("Encoded logits shape:", logits.shape)
print(logits)
decoded = fnspace(logits)
print(decoded)
fnspace.pretty_print()
