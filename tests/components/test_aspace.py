import torch
from loomlib import ActionSpace  # Replace with your actual import
from loomlib.fspace import get_args, TokenSampler, VocabDecoder

# 1. Create an ActionSpace instance
space = ActionSpace()

# 2. Define parameter types
space.angle = VocabDecoder(["left", "right", "center"], temp=1.0)
space.power = TokenSampler(temp=1.0, width=5)

# 3. Define actions with parameters
space.add_action("TURN", {"direction": "angle"})
space.add_action("SHOOT", {"level": "power"})

# 4. Register functions (handlers) for execution
space.TURN = lambda direction: print(f"🚗 Turning {direction}")
space.SHOOT = lambda level: print(f"🎯 Shooting with power {level}")

# 5. Finalize
space.compile()

# 6. Define a sequence of structured actions
sequence = [
    {"type": "TURN", "params": {"direction": "left"}},
    {"type": "SHOOT", "params": {"level": 3}},
]

# 7. Encode to logits
logit_tensor = space.encode(sequence)
print("Encoded tensor shape:", logit_tensor.shape)

# 8. Decode logits back into actions
decoded = space(logit_tensor)
print("Decoded actions:")
for step in decoded:
    print(step)

# 9. Execute the decoded actions
print("\nRunning:")
space.run(decoded)

# 10. Pretty-print the token layout
space.logit_space.pretty_print()
space.logit_space.pretty_print_logits(logit_tensor)

