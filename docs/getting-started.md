# Getting Started

## Installation

```console
$ pip install loomlib
```

## How Loom Works

Loom compiles a Python type-annotated schema into a differentiable output head
(`LoomHead`) that replaces the standard vocabulary projection in a transformer.
The compiler walks your schema, allocates a contiguous range of logits for each
field, wires up type-specific decoders and loss functions, and pre-computes
gradient masks for union branches.

The four examples below progress from a flat model to a tagged union to a full
training loop, then show the encoder side with a round-trip through a
transformer.

---

## Example 1: Flat Model (Beginner)

A `LoomModel` is a flat struct -- each field gets a contiguous slice of the
logit vector. No opcode, no branching.

### Define the schema

```python
from loomlib import LoomModel, Categorical, ContinuousScalar

class SensorReading(LoomModel):
    category: Categorical[3]                   # one-of-3 classification
    temperature: ContinuousScalar[0.0, 100.0]  # bounded continuous value
```

The metaclass introspects the annotations and resolves each one to a `LoomType`
instance. You can inspect what was collected:

```python
>>> SensorReading.loom_fields()
OrderedDict([('category', Categorical(3)),
             ('temperature', ContinuousScalar(0.0, 100.0))])

>>> SensorReading.total_logits()
4  # 3 for category + 1 for temperature
```

### Compile

```python
from loomlib import LoomCompiler

head = LoomCompiler.build_head(SensorReading, d_model=128)
```

The compiler:

1. **Allocates** logit ranges: `category` gets `z[0:3]`, `temperature` gets
   `z[3:4]`.
2. **Stores** the `LoomType` decoder in each `SliceEntry` (no separate
   instantiation step -- types carry their own decode/loss/encode logic).
3. **Generates** gradient masks (trivial here since there's only one branch).

You can visualize the allocation:

```python
>>> print(head.allocation.pretty_print())
SliceAllocation:
  z[0:3]  __root__.category               Categorical(3)
  z[3:4]  __root__.temperature             ContinuousScalar(0.0, 100.0)
  Total logits: 4
```

### Forward, decode, and loss

```python
import torch

hidden = torch.randn(1, 128)   # a single hidden state from a backbone
z = head(hidden)                # Linear(128 -> 4), produces logit vector
assert z.shape == (1, 4)

decoded = head.decode(z)
# decoded["category"]    -> tensor(1)       argmax of softmax over z[0:3]
# decoded["temperature"] -> tensor(52.3)    midpoint + halfrange * tanh(z[3])

targets = {
    "category": torch.tensor([2]),
    "temperature": torch.tensor([60.0]),
}
loss, breakdown = head.loss(z, targets)
# loss = cross_entropy(z[0:3], 2) + mse(decode(z[3:4]), 60.0)
# breakdown = {"category": ..., "temperature": ...}

loss.backward()
```

Each type handles its own loss: `Categorical` uses cross-entropy on raw logits,
`ContinuousScalar` decodes through tanh + affine then applies MSE.

With `loomlib[viz]` installed, you can visualize decoded values across a batch:

```{image} _static/decoded_values.png
:alt: Decoded values for a flat MoveAction model
:width: 100%
```

---

## Example 2: Tagged Union with Gradient Masking (Intermediate)

When an agent can take different *kinds* of actions, use `LoomUnion`. The
compiler prepends an opcode (a `Categorical` softmax over branches) and lays out
each branch's fields sequentially.

### Define the schema

```python
from loomlib import LoomModel, LoomUnion, Categorical, ContinuousScalar, BitInteger

class MoveAction(LoomModel):
    direction: Categorical[4]              # N, S, E, W
    speed: ContinuousScalar[0.0, 10.0]    # bounded scalar

class CastSpell(LoomModel):
    spell_id: Categorical[6]              # spell selection
    power: BitInteger[8]                  # 8-bit power (0--255)

class AgentAction(LoomUnion):
    move: MoveAction
    cast: CastSpell
```

### Compile and inspect

```python
head = LoomCompiler.build_head(AgentAction, d_model=256)
```

The compiler allocates 21 total logits:

```
z[ 0: 2]  __opcode__       Categorical(2)           <- which branch?
z[ 2: 6]  move.direction   Categorical(4)           <- Move fields
z[ 6: 7]  move.speed       ContinuousScalar(0, 10)
z[ 7:13]  cast.spell_id    Categorical(6)           <- Cast fields
z[13:21]  cast.power       BitInteger(8)
```

The opcode is a 2-class softmax: index 0 = `move`, index 1 = `cast`.

```{image} _static/allocation_union.png
:alt: Logit allocation map for AgentAction union
:width: 100%
```

### Gradient masking

During training, only one branch is "active" per sample. Logit gradients for
the inactive branch should be zeroed so they don't receive spurious updates.
The compiler pre-computes binary masks for this:

```python
# Move mask: 1 for opcode + move fields, 0 for cast fields
# [1,1, 1,1,1,1, 1, 0,0,0,0,0,0, 0,0,0,0,0,0,0,0]

# Cast mask: 1 for opcode + cast fields, 0 for move fields
# [1,1, 0,0,0,0, 0, 1,1,1,1,1,1, 1,1,1,1,1,1,1,1]
```

`apply_gradient_mask` uses a custom autograd function that passes the forward
value through unchanged but multiplies gradients by the mask on the backward
pass:

```python
z = head(hidden)
z_masked = head.apply_gradient_mask(z, "move")  # cast logit grads -> 0
loss, breakdown = head.loss(z_masked, targets, active_branch="move")
loss.backward()
```

### Decoding a union

```python
z = head(torch.randn(1, 256))
decoded = head.decode(z)

# decoded["__opcode__"]      -> tensor(0)  means "move"
# decoded["move.direction"]  -> tensor(2)  East
# decoded["move.speed"]      -> tensor(8.3)
# decoded["cast.spell_id"]   -> tensor(4)  (still decoded, just not "active")
# decoded["cast.power"]      -> tensor(37)
```

All branches are always decoded. The opcode tells you which one to *use*.

Here is a heatmap of the raw logit vector for a batch of 6 samples, with field
boundaries marked:

```{image} _static/logits_heatmap.png
:alt: Logit heatmap with field boundaries
:width: 100%
```

### Per-branch loss

```python
targets = {
    "__opcode__": torch.tensor([0]),       # ground truth: Move
    "direction": torch.tensor([0]),        # North
    "speed": torch.tensor([7.5]),
}

z = head(hidden)
z = head.apply_gradient_mask(z, "move")
loss, breakdown = head.loss(z, targets, active_branch="move")
# Only computes loss for opcode + move fields; cast fields are skipped.
```

The loss breakdown across fields, color-coded by type:

```{image} _static/loss_breakdown.png
:alt: Per-field loss breakdown
:width: 100%
```

---

## Example 3: Full Training Loop (Advanced)

This example puts everything together: native Python types, an optimizer loop,
re-entrant encoding, post-decode function composition, and constraint penalties.

### Schema with native types

Loom resolves bare Python type annotations to sensible defaults:

| Python type | Loom type | Logits |
|-------------|-----------|--------|
| `bool` | `Boolean` | 1 |
| `int` | `BitInteger[32]` | 32 |
| `float` | `Scalar` | 1 |

You can mix them freely with explicit Loom types:

```python
from loomlib import LoomModel, LoomUnion, LoomCompiler, Categorical

class Attack(LoomModel):
    target_id: Categorical[8]
    critical: bool                # -> Boolean, 1 logit
    damage: float                 # -> Scalar, 1 logit (unbounded)

class Heal(LoomModel):
    amount: int                   # -> BitInteger[32], 32 logits

class Action(LoomUnion):
    attack: Attack
    heal: Heal

head = LoomCompiler.build_head(Action, d_model=64)
# opcode(2) + target_id(8) + critical(1) + damage(1) + amount(32) = 44
assert head.total_logits == 44
```

### Training loop

```python
import torch

optimizer = torch.optim.Adam(head.parameters(), lr=1e-3)
backbone = torch.nn.Linear(32, 64)  # stand-in for a transformer backbone

for step in range(200):
    hidden = backbone(torch.randn(8, 32))
    z = head(hidden)

    # Suppose samples 0-3 are attacks, 4-7 are heals
    targets = {
        "__opcode__": torch.tensor([0, 0, 0, 0, 1, 1, 1, 1]),
        "target_id": torch.tensor([1, 3, 0, 7, 0, 0, 0, 0]),
        "critical": torch.tensor([1, 0, 0, 1, 0, 0, 0, 0], dtype=torch.float),
        "damage": torch.tensor([5.0, 2.3, 1.1, 9.9, 0, 0, 0, 0]),
        "amount": torch.tensor([0, 0, 0, 0, 50, 100, 75, 200]),
    }

    z_attack = head.apply_gradient_mask(z, "attack")
    loss_attack, _ = head.loss(z_attack, targets, active_branch="attack")

    z_heal = head.apply_gradient_mask(z, "heal")
    loss_heal, _ = head.loss(z_heal, targets, active_branch="heal")

    loss = loss_attack + loss_heal
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

### Re-entrant encoding

`encode_action` maps a decoded action back into embedding space. This is useful
for autoregressive architectures where the previous action feeds back as input:

```python
z = head(torch.randn(1, 64))
decoded = head.decode(z)

embedding = head.encode_action(decoded)  # (1, 64) -- back to d_model
assert embedding.shape[-1] == 64
```

Under the hood, each `LoomType.encode()` converts the value back to a
logit-sized tensor, all entries are concatenated, and a learned linear layer
projects the result to `d_model`.

### Post-decode composition with FnSpace

`FnSpace` lets you register functions that consume decoded values and produce
derived quantities. Functions are evaluated in registration order, and later
functions can reference earlier outputs by name:

```python
from loomlib import LoomModel, LoomCompiler, ContinuousScalar
from loomlib.fn import FnSpace

class Physics(LoomModel):
    mass: ContinuousScalar[0.1, 100.0]
    velocity: ContinuousScalar[-10.0, 10.0]

head = LoomCompiler.build_head(Physics, d_model=64)

fn = FnSpace()
fn.add_fn("momentum", lambda mass, velocity: mass * velocity)
fn.add_fn("kinetic_energy", lambda mass, velocity: 0.5 * mass * velocity ** 2)

z = head(torch.randn(1, 64))
decoded = head.decode(z)
results = fn.evaluate(decoded)
# results["mass"], results["velocity"], results["momentum"], results["kinetic_energy"]
```

### Constraint penalties

The `fn` module also provides differentiable penalty functions you can add to
the loss:

```python
from loomlib.fn import less_than_penalty, range_penalty

z = head(torch.randn(1, 64))
decoded = head.decode(z)

# Penalize if velocity is outside [-5, 5]
penalty = range_penalty(decoded["velocity"], -5.0, 5.0)

# Penalize if mass > velocity (arbitrary constraint for illustration)
penalty += less_than_penalty(decoded["mass"], decoded["velocity"]).sum()

total_loss = supervised_loss + 0.1 * penalty
total_loss.backward()
```

`less_than_penalty(a, b)` returns `ReLU(a - b)` (zero when `a < b`).
`range_penalty(x, lo, hi)` returns `ReLU(lo - x) + ReLU(x - hi)`.

### Custom types

To create a new type, subclass `LoomType` and implement four methods:

```python
from loomlib.types import LoomType
import torch
import torch.nn.functional as F

class UnitVector(LoomType):
    """Decodes logits to a unit vector via L2 normalization."""

    def __init__(self, dim: int):
        self._dim = dim

    def logit_size(self) -> int:
        return self._dim

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return F.normalize(z, dim=-1)

    def loss(self, z: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        predicted = self.decode(z)
        return 1.0 - F.cosine_similarity(predicted, target, dim=-1).mean()

    def encode(self, value, device=None):
        device = device or torch.device("cpu")
        if not isinstance(value, torch.Tensor):
            value = torch.tensor(value, dtype=torch.float32, device=device)
        return value.to(device)
```

Then use it in a schema like any other type:

```python
class Steering(LoomModel):
    heading: UnitVector(3)           # pass an instance directly
    throttle: ContinuousScalar[0.0, 1.0]
```

---

## Example 4: Encoding Structured Input

The previous examples showed the *decode* side -- turning transformer output
into structured values. `LoomEncoder` handles the *encode* side: turning
structured instances into field-level tokens that feed into the transformer.
Both are compiled from the same schema.

### Define a schema with scalar-encoding types

Encoder fields must use types whose `encode()` method returns a single scalar.
The built-in `ContinuousScalar`, `Boolean`, and `Scalar` types all satisfy this:

```python
from loomlib import (
    LoomModel, LoomUnion, LoomCompiler,
    ContinuousScalar, Boolean, Scalar,
)

class Steering(LoomModel):
    angle: ContinuousScalar[-1.0, 1.0]
    throttle: ContinuousScalar[0.0, 1.0]

class Shooting(LoomModel):
    active: Boolean()
    power: Scalar()

class GameAction(LoomUnion):
    steer: Steering
    shoot: Shooting
```

### Build the encoder

```python
encoder = LoomCompiler.build_encoder(GameAction, d_model=64)
```

### Collate instances into a LoomBatch

The `collate()` method accepts a list of sequences, where each sequence is a
list of `(branch_name, {field: value})` tuples:

```python
data = [
    [("steer", {"angle": 0.5, "throttle": 0.8}),
     ("shoot", {"active": True, "power": 3.0}),
     ("steer", {"angle": -0.2, "throttle": 0.1})],
    [("shoot", {"active": False, "power": -1.0})],
]
batch = encoder.collate(data)
```

The result is a `LoomBatch` with five columnar tensors. Sequence 0 has 6 tokens
(2 + 2 + 2 fields), while sequence 1 has 2; the rest is right-padded.

```{image} _static/encoder_batch.png
:alt: Field-token batch layout for a GameAction union
:width: 100%
```

### Forward pass

```python
embeddings = encoder(batch)        # [2, 6, 64]
assert embeddings.shape == (2, 6, 64)
```

### Wiring into a transformer round-trip

The `padding_mask` follows PyTorch's `src_key_padding_mask` convention
(`True` = padding), so it plugs directly into `nn.TransformerEncoder`:

```python
import torch.nn as nn

transformer = nn.TransformerEncoder(
    nn.TransformerEncoderLayer(d_model=64, nhead=4, batch_first=True),
    num_layers=2,
)
hidden = transformer(embeddings, src_key_padding_mask=batch.padding_mask)

head = LoomCompiler.build_head(GameAction, d_model=64)
z = head(hidden[:, 0, :])    # use first token's hidden state
decoded = head.decode(z)
```

### Custom branch embeddings

To override how a specific branch embeds its fields, pass a custom module:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyShootEmbed(nn.Module):
    def __init__(self, num_fields, d_model):
        super().__init__()
        self.proj = nn.Linear(num_fields, d_model)

    def forward(self, field_ids, values):
        one_hot = F.one_hot(field_ids, num_classes=2).float()
        return self.proj(one_hot * values.unsqueeze(-1))

encoder = LoomCompiler.build_encoder(
    GameAction, d_model=64,
    branch_embeddings={"shoot": MyShootEmbed(2, 64)},
)
```

Branches without an explicit module use the `DefaultBranchEmbedding`, which
combines a learned field identity with a value-scaled embedding.
