# Loom

> Weaving typed action spaces from transformer logits

[![PyPI](https://img.shields.io/pypi/v/loomlib.svg)](https://pypi.python.org/pypi/loomlib/)
[![Python](https://img.shields.io/pypi/pyversions/loomlib.svg)](https://pypi.python.org/pypi/loomlib/)
![Build](https://github.com/btrainwilson/loom/actions/workflows/test.yaml/badge.svg)

Loom replaces the monolithic vocabulary projection in transformer models with
structured, typed codec layers. A single forward pass can simultaneously predict
categorical decisions, continuous values, and composite structured actions --
each with its own type-aware loss function. On the input side, structured
observations are encoded into embeddings via matched type-aware projections.
You define a schema once with Python type annotations, and the compiler generates
both sides -- logit allocation, decoding, encoding, gradient masking, and loss
composition -- automatically.

## Install

```bash
pip install loomlib
```

## Quick Start

Define a schema, compile it, and use it:

```python
import torch
from loomlib import LoomModel, LoomCompiler, Categorical, ContinuousScalar

class SensorReading(LoomModel):
    category: Categorical[3]                   # 3 logits -> softmax
    temperature: ContinuousScalar[0.0, 100.0]  # 1 logit  -> tanh + affine

head = LoomCompiler.build_head(SensorReading, d_model=128)

z = head(torch.randn(1, 128))          # project hidden state to 4 logits
decoded = head.decode(z)               # {"category": tensor(1), "temperature": tensor(52.3)}

targets = {"category": torch.tensor([2]), "temperature": torch.tensor([60.0])}
loss, breakdown = head.loss(z, targets) # composite cross-entropy + MSE
loss.backward()
```

For tagged unions, gradient masking, training loops, and more, see the
[full documentation](https://loomlib.readthedocs.io).

## Architecture

Loom organizes the model interface as a **typed codec** -- a matched
encoder-decoder pair compiled from a single schema:

```
              ENCODER (input)                         DECODER (output)
        structured observation                      hidden state (d_model)
                 |                                         |
                 v                                         v
        [Type Encoders]                             [Slice Layer]
         per-field learned                           partition logit vector
         projections                                 into typed sub-vectors
                 |                                         |
                 v                                         v
        [Sum Pool + LayerNorm]                      [Type Layer]
         combine into single                         apply type-specific
         embedding vector                            decoders + losses
                 |                                         |
                 v                                         v
          embedding (d_model)                       [Fn Layer]
                 |                                   compose into function calls
                 v                                         |
           transformer                                     v
            backbone                                [Action Layer]
                 |                                   dispatch via opcode
                 v
          hidden state (d_model) ──────────────────> DECODER
```

Both sides are compiled from the **same schema**, guaranteeing that
representational assumptions are consistent across the input-output boundary.

## Typed Codec

The core idea: every type knows how to **decode** (logits → value), **encode**
(value → embedding), and compute its own **loss**.

### Decoding (output side)

The output logit vector is partitioned into typed slices. Each slice is decoded
by a type-appropriate activation and trained with a type-appropriate loss:

```python
head = LoomCompiler.build_head(AgentAction, d_model=256)
z = head(hidden_state)                    # project to logit space
z = head.apply_gradient_mask(z, "move")   # mask inactive branches
loss, breakdown = head.loss(z, targets)   # composite typed loss
```

### Encoding (input side)

Structured observations are encoded into the model's embedding space via
type-aware learned projections -- no tokenization or flattening required:

```python
codec = LoomCompiler.build_codec(AgentAction, d_model=256)

# Encode a structured observation directly
obs = {"direction": 2, "speed": 7.5, "is_sprinting": True}
embedding = codec.encoder(obs)   # -> tensor of shape (1, 256)

# Feed through transformer, then decode
hidden = transformer(embedding)
action = codec.decoder.decode(codec.decoder(hidden))
```

Each type uses a semantically meaningful encoding:

| Type | Encoder | What it preserves |
|------|---------|-------------------|
| `Categorical[n]` | Learned embedding table | Discrete identity |
| `ContinuousScalar[lo, hi]` | Learned linear from normalized value | Magnitude and ordering |
| `BitInteger[B]` | Bit decomposition → learned linear | Place-value structure |
| `OrdinalInteger[K]` | Thermometer → learned linear | Ordinal relationships |
| `Boolean` | Learned 2-entry embedding | Binary distinction |

For composite inputs, field embeddings are combined via sum pooling + layer norm
into a single vector per timestep. This maintains a one-to-one correspondence
between environment steps and model timesteps, regardless of how many typed
fields the observation contains.

**Why this matters:** Standard approaches either tokenize everything (losing
numeric precision and wasting sequence length) or flatten to a raw vector
(losing type semantics). Loom's typed encoder preserves the structure: an
integer 128 is embedded near 129, a categorical "North" is embedded distinctly
from "South", and a continuous 3.7 carries its magnitude into the embedding
directly.

## Built-in Types

| Type | Logits | Decoder | Encoder | Loss |
|------|--------|---------|---------|------|
| `Categorical[n]` | n | Softmax | Embedding table | Cross-Entropy |
| `ContinuousScalar[lo, hi]` | 1 | Tanh → affine | Learned linear | MSE |
| `BitInteger[B]` | B | Sigmoid per bit | Bit decomposition → linear | BCE + γ·MSE |
| `OrdinalInteger[K]` | K | Cumulative sigmoid | Thermometer → linear | BCE (per threshold) |
| `Boolean` | 1 | Sigmoid | Learned 2-entry | BCE |
| `Scalar` | 1 | Identity | Learned linear | MSE |

Native Python types are also supported: `bool` maps to `Boolean`, `int` to
`BitInteger[32]`, and `float` to `Scalar`.

**BitInteger vs OrdinalInteger:** BitInteger encodes values in base-2 with
sigmoid activations per bit. OrdinalInteger uses cumulative thresholds
(thermometer encoding), which naturally penalizes large errors more than small
ones. BitInteger is more compact (⌈log₂K⌉ logits vs K), while OrdinalInteger
has better gradient properties for ordinal data. Both are available; we
recommend ablating for your domain.

Custom types can be added by subclassing `LoomType` and implementing
`logit_size()`, `decode()`, `encode()`, and `loss()`.

## Type Parser (Coming Soon)

Loom can auto-generate schemas from existing type definitions:

```python
from loomlib import LoomParser

# From a Python function signature
schema = LoomParser.from_function(my_api_call)

# From a Gymnasium action space
schema = LoomParser.from_gym_space(env.action_space)

# From a JSON Schema / OpenAPI spec
schema = LoomParser.from_json_schema(api_spec)

# From example data (type inference)
schema = LoomParser.from_data({"speed": [0.5, 1.2, ...], "mode": ["walk", "run", ...]})

# Then compile as usual
codec = LoomCompiler.build_codec(schema, d_model=256)
```

## Citation

```bibtex
@inproceedings{wilson2026loom,
  title={Loom: Well-Typed Action Spaces for Transformer Logits},
  author={Wilson, Blake A.},
  year={2026}
}
```

## License

MIT
