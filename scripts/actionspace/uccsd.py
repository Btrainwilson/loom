from loomlib import ActionSpace
from loomlib.thunker import VocabDecoder

from dataclass import dataclass


# 1. Create action space and add types (BEFORE compile)
pv = ActionSpace()

pv.actions = [] 

pv.qubit = VocabDecoder(list(range(10)), temp=1.0)
pv.angle = VocabDecoder([2**(-i) / 320 for i in range(-5, 5)], temp=1.0)

pv.SE = DoubleExcitation(q1=pv.qubit,
               q2=pv.qubit,
               q3=pv.qubit,
               q4=pv.qubit,
               angle=pv.angle)
pv.DE = Action


# Add actions. Handler args must match the parameter names.
pv.add_action(
    "SE",
    {"q1": "qubit", "q2": "qubit", "a": "angle"},
    handler=lambda q1, q2, a: print("SE:", q1, q2, a )
)
pv.add_action(
    "DE",
    {"q1": "qubit", "q2": "qubit", "q3": "qubit", "q4": "qubit", "a": "angle"},
    handler=lambda q1, q2, q3, q4, a: print("DE:", q1, q2, q3, q4, a * 3.14 / 10)
)

# COMPILE before encode/decode
pv.compile()

# 2. Encode a sequence
tokens = pv.encode([
    {"type": "SE", "params": {"q1": 2, "q2": 6, "a": 1 }},
    {"type": "SE", "params": {"q1": 3, "q2": 8, "a": 2 }},
    {"type": "DE", "params": {"q1": 1, "q2": 4, "q3": 6, "q4": 9, "a": 5 }},
])
print("Tokens:", tokens)

# 3. Decode
decoded = pv(tokens)
print("Decoded:", decoded)

# 4. Run
pv.run(decoded)

# 5. Vocab print (if supported)
if hasattr(pv, 'print_vocab'):
    pv.print_vocab()

pv.print_run(decoded)

