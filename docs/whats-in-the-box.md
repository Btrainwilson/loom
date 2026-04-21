# Architecture

Loom provides schema-driven structured I/O for transformers. A single schema
compiles into both an encoder (input tokenization) and a decoder head (output
projection), with a compiler that wires them together. The system is organized
into four decoding layers, an encoding layer, and two `nn.Module` endpoints.

```
                             Schema (LoomModel / LoomUnion)
                                        |
                                        v
                               ┌─────────────────┐
                               │  Loom Compiler   │
                               │  (3-phase build) │
                               └────────┬────────┘
                                        |
                    ┌───────────────────┬┴──────────────────┐
                    v                   v                   v
             SliceAllocation     Gradient Masks      Branch Fields
                    |                   |                   |
                    └────────┬──────────┘                   |
                             v                              v
                        ┌──────────┐                 ┌─────────────┐
                        │ LoomHead │ nn.Module       │ LoomEncoder │ nn.Module
                        └──────────┘                 └─────────────┘
                 forward / decode / loss       collate / forward (embed)
```

---

## Schema Layer

### LoomModel

A flat struct. Subclass it and annotate fields with Loom types:

```python
class MoveAction(LoomModel):
    direction: Categorical[4]
    speed: ContinuousScalar[0.0, 10.0]
```

The `LoomModelMeta` metaclass reads `__annotations__`, resolves each annotation
to a `LoomType` instance, and stores the result in `cls._loom_fields` (an
`OrderedDict`). Fields starting with `_` are skipped. Native Python types
(`bool`, `int`, `float`) are resolved via a default mapping.

**Key methods:** `loom_fields()`, `total_logits()`.

### LoomUnion

A tagged union over `LoomModel` branches:

```python
class AgentAction(LoomUnion):
    move: MoveAction
    cast: CastSpell
```

`LoomUnionMeta` requires every annotation to be a `LoomModel` subclass. The
compiler prepends an opcode slice (a `Categorical` over the number of branches)
so the model can select which branch is active.

**Key methods:** `loom_branches()`, `num_branches()`, `branch_names()`.

---

## Layer 1: Slice Layer

**Module:** `loomlib.slice.allocation`

The slice layer partitions a flat logit vector into named, typed sub-vectors.
Here is an example allocation for a union with two branches (21 total logits):

```{image} _static/allocation_union.png
:alt: Logit allocation map for a union schema
:width: 100%
```

### SliceEntry

A single allocated region:

| Field | Description |
|-------|-------------|
| `name` | Fully-qualified name, e.g. `"move.direction"` or `"__opcode__"` |
| `start` | Start index (inclusive) |
| `end` | End index (exclusive) |
| `thunk_type` | The `LoomType` instance that decodes this slice |

`entry.extract(z)` returns `z[..., start:end]`, handling arbitrary batch
dimensions.

### BranchAllocation

Groups all `SliceEntry` objects belonging to one branch of a union (or the
single `__root__` branch of a flat model).

### SliceAllocation

The complete allocation tree:

- `opcode: SliceEntry | None` -- present only for unions.
- `branches: dict[str, BranchAllocation]` -- one per union branch (or a single
  `__root__` for flat models).
- `total_logits: int` -- total width of the logit vector.
- `build_gradient_mask(branch)` -- returns a binary tensor with 1s for the
  opcode + the named branch, 0s elsewhere.
- `pretty_print()` -- human-readable allocation table.
- `all_entries()` -- flat list of every `SliceEntry` in allocation order.

---

## Layer 2: Type Layer

**Module:** `loomlib.types`

Every field type is a `LoomType` subclass that implements four methods:

| Method | Purpose |
|--------|---------|
| `logit_size()` | Number of logits consumed |
| `decode(z)` | Map raw logits to a typed value |
| `loss(z, target)` | Type-aware loss between logits and ground truth |
| `encode(value)` | Convert a value back to logit-shaped tensor |

### Built-in types

**Categorical(n)**

Softmax classification over `n` classes. `decode` returns `argmax`, `loss` uses
`F.cross_entropy`. Also provides `decode_probs()` for the full probability
vector and `log_prob()` for policy-gradient methods.

**ContinuousScalar(lo, hi)**

Maps a single logit to `[lo, hi]` via `midpoint + halfrange * tanh(z)`. Loss is
MSE between the decoded value and target. `encode` uses `atanh` to invert.

**BitInteger(B)**

Represents an integer in `[0, 2^B - 1]` as `B` independent sigmoid bits. Loss
combines bitwise BCE with a magnitude-level MSE term (weighted by `gamma`,
default 0.1). `decode_soft()` provides a differentiable relaxation.

**Boolean**

Single sigmoid logit, thresholded at 0.5. Encodes `True` as `+6.0`, `False` as
`-6.0` (deep in the sigmoid's saturation region).

**Scalar**

Unbounded identity pass-through. MSE loss. This is the default for bare `float`
annotations.

### Custom types

Subclass `LoomType` and implement the four abstract methods. The compiler will
pick up any `LoomType` instance found in a schema annotation.

---

## Layer 3: Fn Layer

**Module:** `loomlib.fn`

The Fn layer composes decoded values into derived quantities via registered
Python functions.

### FnSpace

```python
fn = FnSpace()
fn.add_fn("momentum", lambda mass, velocity: mass * velocity)
```

`evaluate(decoded_values)` runs each function in registration order. Argument
names are matched to keys in the decoded-values dict (or to outputs of earlier
functions), so functions can chain:

```python
fn.add_fn("double_x", lambda x: x * 2)
fn.add_fn("sum", lambda double_x, y: double_x + y)
# evaluate({"x": 3, "y": 1}) -> {"x": 3, "y": 1, "double_x": 6, "sum": 7}
```

### Constraint penalties

Three differentiable penalty helpers:

| Function | Formula | Zero when |
|----------|---------|-----------|
| `less_than_penalty(a, b)` | `ReLU(a - b)` | `a < b` |
| `range_penalty(x, lo, hi)` | `ReLU(lo - x) + ReLU(x - hi)` | `lo <= x <= hi` |
| `equality_penalty(a, b)` | `(a - b)^2` | `a == b` |

Add these to the loss to enforce soft constraints on decoded values.

---

## Layer 4: Action Layer

**Module:** `loomlib.action.dispatch`

`ActionDispatch` routes a logit vector to the correct union branch:

- `select_action(z)` -- argmax over the opcode logits, returns branch indices
  and names for each sample in the batch.
- `decode_branch(z, branch_name)` -- decode only the fields belonging to one
  branch.
- `action_probs(z)` -- softmax probabilities over branches (useful for
  policy-gradient RL).

For flat models (no union), `select_action` returns the single `__root__`
branch.

---

## The Compiler

**Module:** `loomlib.compiler`

`LoomCompiler.build_head(schema, d_model)` runs a three-phase pipeline:

### Phase 1: Allocation

Walks the schema tree and assigns contiguous logit index ranges.

- **Flat model:** single `__root__` branch. Fields are laid out sequentially
  starting at offset 0.
- **Union:** an opcode slice (sized by number of branches) is allocated first,
  then each branch's fields are appended sequentially.

The result is a `SliceAllocation` object.

### Phase 2: Decoder graph

No separate instantiation step. Each `LoomType` is already stored in its
`SliceEntry` during allocation and carries its own `decode`, `loss`, and
`encode` logic.

### Phase 3: Mask generation

For each branch in the allocation, a binary gradient mask is pre-computed:
`1` for the opcode slice + that branch's field slices, `0` elsewhere. These
masks are stored as non-trainable parameters in the `LoomHead`.

---

## LoomHead

**Module:** `loomlib.head`

`LoomHead` is the `nn.Module` produced by the compiler. It owns:

- `projection: Linear(d_model, total_logits)` -- the logit projection.
- `encoder_projection: Linear(total_logits, d_model)` -- for re-entrant
  encoding.
- `_gradient_masks: ParameterDict` -- pre-computed masks (non-trainable).

### Key methods

**`forward(hidden)`**

Projects hidden states to logit space. Accepts `(batch, d_model)` or
`(batch, seq, d_model)`.

**`decode(z)`**

Walks every `SliceEntry` and calls `thunk_type.decode()`. Returns a dict keyed
by field name (flat models) or `branch.field` (unions), plus `__opcode__` for
unions.

**`loss(z, targets, active_branch=None, loss_weights=None)`**

Computes composite loss. Each field's loss is computed by its `LoomType` and
summed. `active_branch` restricts computation to one branch. `loss_weights`
scales individual field losses. Returns `(total_loss, breakdown_dict)`.

**`apply_gradient_mask(z, branch)`**

Identity on the forward pass; on backward, multiplies `grad_output` by the
branch's binary mask. This zeros gradients for logit dimensions outside the
opcode + active branch.

**`encode_action(decoded)`**

Each field's value is re-encoded via `LoomType.encode()`, concatenated into a
full logit-sized vector, and projected back to `d_model` through
`encoder_projection`. Useful for autoregressive action feedback.

---

## LoomEncoder

**Module:** `loomlib.encoder`

`LoomEncoder` is the input-side counterpart to `LoomHead`. Where `LoomHead`
decodes transformer output logits into structured values, `LoomEncoder`
tokenizes structured input instances into field-level embeddings that feed
*into* the transformer. Both are constructed from the same schema.

### Construction

```python
encoder = LoomCompiler.build_encoder(schema, d_model=128)
```

The compiler walks the schema, builds a global field-ID map, and instantiates
per-branch embedding modules.

### LoomBatch

`LoomBatch` is a frozen dataclass holding five columnar tensors for a batch of
field-tokenized instances:

| Tensor | Shape | Dtype | Description |
|--------|-------|-------|-------------|
| `type_ids` | `[B, N_max]` | long | Branch index for each token |
| `inst_ids` | `[B, N_max]` | long | Instance index within the sequence |
| `field_ids` | `[B, N_max]` | long | Global field index |
| `values` | `[B, N_max]` | float | Encoded scalar value (one per token) |
| `padding_mask` | `[B, N_max]` | bool | `True` = padding, `False` = real (PyTorch convention) |

### collate(instances)

Converts raw Python data into a `LoomBatch`. The input format is a nested list:

```python
instances = [
    # sequence 0: two instances
    [("steer", {"angle": 0.5, "throttle": 0.8}),
     ("shoot", {"active": True, "power": 3.0})],
    # sequence 1: one instance
    [("steer", {"angle": -0.2, "throttle": 0.1})],
]
batch = encoder.collate(instances)
```

Each field value is passed through its `LoomType.encode()` to produce a single
scalar. Sequences shorter than the longest are right-padded.

### forward(batch)

Embeds a `LoomBatch` into `[B, N_max, d_model]`. For each branch, the encoder:

1. Selects tokens belonging to that branch via `type_ids`.
2. Passes their `field_ids` and `values` to the branch's embedding module.
3. Adds shared type and instance-positional embeddings.

Padding positions remain zero.

### DefaultBranchEmbedding

The default per-branch embedding is additive: a learned field identity embedding
plus a value-scaled embedding.

```python
out = field_embedding(field_ids) + values.unsqueeze(-1) * value_scale(field_ids)
```

Every branch embedding module must follow this contract:

```python
forward(field_ids: Tensor[N], values: Tensor[N]) -> Tensor[N, d_model]
```

where `field_ids` are **0-based within the branch**.

### Custom branch embeddings

To override the embedding for a specific branch, pass a dict to `build_encoder`:

```python
class MyShootEmbed(nn.Module):
    def __init__(self, num_fields, d_model):
        super().__init__()
        self.proj = nn.Linear(num_fields, d_model)

    def forward(self, field_ids, values):
        one_hot = F.one_hot(field_ids, num_classes=2).float()
        return self.proj(one_hot * values.unsqueeze(-1))

encoder = LoomCompiler.build_encoder(
    GameAction, d_model=128,
    branch_embeddings={"shoot": MyShootEmbed(2, 128)},
)
```

Branches without an explicit module get the `DefaultBranchEmbedding`.
