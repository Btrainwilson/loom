import torch
from loomlib import ActionSpace  # Replace with your actual import

# Dummy type function classes for demonstration
class IdentityType:
    def __len__(self): return 1
    def decode(self, x): return x.item() if isinstance(x, torch.Tensor) and x.numel() == 1 else x
    def encode(self, value, device=None): return torch.tensor([float(value)], device=device or "cpu")

class IntType(IdentityType): pass

def add_handler(x, y):
    print(f"add_handler called: x={x}, y={y}")
    return x + y

def mul_handler(x, y):
    print(f"mul_handler called: x={x}, y={y}")
    return x * y

def main():
    # 1. Create ActionSpace and add types (parameters)
    action_space = ActionSpace()
    action_space.add_type(x=IntType(), y=IntType())

    # 2. Register actions and handlers
    action_space.add_action("add", {"x": "x", "y": "y"}, add_handler)
    action_space.add_action("mul", {"x": "x", "y": "y"}, mul_handler)
    action_space.compile()

    # 3. Generate random logits
    # Shape: [batch, width]
    batch = 3
    torch.manual_seed(42)
    logits = torch.randn(batch, action_space.width)
    print("\nLogits tensor:")
    print(logits)

    # 4. Replace logits so that each row's action is add/mul/STOP, and x/y are easy values
    logits.zero_()
    for i, act in enumerate(["add", "mul", "STOP"]):
        # Set action token as a one-hot
        action_idx = action_space.action_decoder.choices.index(act)
        a_slice = action_space.fspace.type_indices["action"]
        logits[i, a_slice] = torch.nn.functional.one_hot(torch.tensor(action_idx), len(action_space.action_decoder)).float()
        # Set x and y
        logits[i, action_space.fspace.type_indices["x"]] = float(i+1)
        logits[i, action_space.fspace.type_indices["y"]] = float((i+1)*2)

    print("\nLogits after forced setup:")
    print(logits)

    # 5. Decode logits to actions/params
    decoded_seq = action_space(logits)
    print("\nDecoded sequence:")
    for step in decoded_seq:
        print(step)

    # 6. Encode back to logits
    reencoded = action_space.encode(decoded_seq)
    print("\nRe-encoded tensor (should have the same shape as input):")
    print(reencoded)

    # 7. Run the sequence
    print("\nRunning decoded sequence:")
    action_space.run(decoded_seq)

    # 8. Print run with pretty output
    print("\nPrint run:")
    action_space.print_run(decoded_seq)

if __name__ == "__main__":
    main()
