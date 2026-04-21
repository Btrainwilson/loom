# Loom

> Well-typed structured I/O for transformers

[![PyPI](https://img.shields.io/pypi/v/loomlib.svg)](https://pypi.python.org/pypi/loomlib/)
[![Python](https://img.shields.io/pypi/pyversions/loomlib.svg)](https://pypi.python.org/pypi/loomlib/)
![Build](https://github.com/btrainwilson/loom/actions/workflows/test.yaml/badge.svg)

Loom provides schema-driven encoding and decoding for transformers operating on
structured data. A single type-annotated schema drives both sides: a
`LoomEncoder` that tokenizes structured instances into field-level embeddings
for the transformer's input, and a `LoomHead` that decodes the transformer's
output logits back into typed values with matching loss functions. The schema is
the contract -- it guarantees type-correct structured I/O on both sides of the
transformer.

**One schema, two modules.** The same `LoomModel` or `LoomUnion` definition
compiles into both an encoder and a decoder head.

**Field-level tokenization.** The encoder makes each field its own token,
giving the transformer per-field attention over structured inputs.

**Type-safe round-trip.** Structured data goes in, the transformer processes
it, structured data comes out -- typed end to end.

**Composable action spaces.** Unions let you represent heterogeneous action
types (move, cast, heal) with branching and per-branch gradient masking on the
decode side, and per-branch embedding modules on the encode side.

```python
from loomlib import LoomModel, LoomUnion, LoomCompiler, Categorical, ContinuousScalar

class MoveAction(LoomModel):
    direction: Categorical[4]
    speed: ContinuousScalar[0.0, 10.0]

encoder = LoomCompiler.build_encoder(MoveAction, d_model=128)
head = LoomCompiler.build_head(MoveAction, d_model=128)
```

```{toctree}
:maxdepth: 2

getting-started.md
whats-in-the-box.md
visualization.md
code.md
contributing.md
```
