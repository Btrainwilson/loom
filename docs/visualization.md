# Visualization

Loom ships an optional `viz` subpackage with opinionated matplotlib helpers for
inspecting schemas, logit vectors, losses, and decoded outputs. Install with:

```console
$ pip install loomlib[viz]
```

All functions accept Loom objects directly and return a
`matplotlib.figure.Figure`. Pass an optional `ax` keyword argument to draw into
an existing axes for custom layouts.

```python
from loomlib.viz import (
    plot_allocation, plot_logits, plot_loss_breakdown, plot_decoded,
    plot_batch, plot_token_grid, plot_embedding_norms,
)
```

## Decoder Visualizations

---

## plot_allocation

Draws a horizontal stacked-bar chart of the logit vector, partitioned by branch.
Each `SliceEntry` is a colored rectangle labeled with the field name and type.

Accepts a `LoomHead` or a `SliceAllocation`.

```python
from loomlib import LoomModel, LoomUnion, LoomCompiler, Categorical, ContinuousScalar, BitInteger
from loomlib.viz import plot_allocation

class MoveAction(LoomModel):
    direction: Categorical[4]
    speed: ContinuousScalar[0.0, 10.0]

class CastSpell(LoomModel):
    spell_id: Categorical[6]
    power: BitInteger[8]

class AgentAction(LoomUnion):
    move: MoveAction
    cast: CastSpell

head = LoomCompiler.build_head(AgentAction, d_model=256)
fig = plot_allocation(head)
```

```{image} _static/allocation_union.png
:alt: Logit allocation map for AgentAction union
:width: 100%
```

---

## plot_logits

Heatmap of raw logit values with field boundaries overlaid as dashed lines.
Uses a diverging colormap centered at zero so positive and negative logits are
visually distinct.

```python
import torch
from loomlib.viz import plot_logits

z = head(torch.randn(6, 256))
fig = plot_logits(head, z)
```

```{image} _static/logits_heatmap.png
:alt: Logit heatmap with field boundaries
:width: 100%
```

---

## plot_loss_breakdown

Horizontal bar chart of per-field losses, sorted by magnitude. Pass the
optional `allocation` argument to color-code bars by `LoomType`.

```python
from loomlib.viz import plot_loss_breakdown

targets = {
    "__opcode__": torch.tensor([0, 0, 0, 1, 1, 1]),
    "direction": torch.tensor([0, 1, 2, 0, 0, 0]),
    "speed": torch.tensor([5.0, 3.0, 7.5, 0.0, 0.0, 0.0]),
    "spell_id": torch.tensor([0, 0, 0, 3, 5, 1]),
    "power": torch.tensor([0, 0, 0, 100, 200, 50]),
}

_, breakdown = head.loss(z, targets)
fig = plot_loss_breakdown(breakdown, allocation=head.allocation)
```

```{image} _static/loss_breakdown.png
:alt: Per-field loss breakdown bar chart
:width: 100%
```

---

## plot_decoded

Per-field subplots that adapt to each field's type:

- **Categorical** -- bar chart of class counts across the batch.
- **Boolean** -- stacked proportion bar (True / False).
- **Continuous** (`ContinuousScalar`, `Scalar`, `BitInteger`) -- strip plot of
  decoded values, with the type's range shaded when bounded.

```python
from loomlib import LoomCompiler
from loomlib.viz import plot_decoded

flat_head = LoomCompiler.build_head(MoveAction, d_model=128)
z_flat = flat_head(torch.randn(20, 128))
fig = plot_decoded(flat_head, z_flat)
```

```{image} _static/decoded_values.png
:alt: Decoded values per field
:width: 100%
```

---

## Encoder Visualizations

The following functions visualize the *input* side of the Loom pipeline --
collated batches and encoder embeddings.

### plot_batch

Draws a per-sequence row of colored field-token blocks. Each real token is a
rectangle colored by branch, labeled with the field name and encoded scalar
value. Padding tokens are gray. Instance boundaries are marked with dividers.

```python
from loomlib import LoomModel, LoomUnion, LoomCompiler, ContinuousScalar, Boolean, Scalar
from loomlib.viz import plot_batch

class Steering(LoomModel):
    angle: ContinuousScalar[-1.0, 1.0]
    throttle: ContinuousScalar[0.0, 1.0]

class Shooting(LoomModel):
    active: Boolean()
    power: Scalar()

class GameAction(LoomUnion):
    steer: Steering
    shoot: Shooting

encoder = LoomCompiler.build_encoder(GameAction, d_model=64)
data = [
    [("steer", {"angle": 0.5, "throttle": 0.8}),
     ("shoot", {"active": True, "power": 3.0}),
     ("steer", {"angle": -0.2, "throttle": 0.1})],
    [("shoot", {"active": False, "power": -1.0})],
]
batch = encoder.collate(data)
fig = plot_batch(encoder, batch)
```

```{image} _static/encoder_batch.png
:alt: Field-token batch layout
:width: 100%
```

---

### plot_token_grid

Heatmap of encoded scalar values across the batch. Rows are sequences, columns
are token positions. Padding is masked to gray. Instance boundaries from the
first sequence are overlaid as dashed lines.

```python
from loomlib.viz import plot_token_grid

fig = plot_token_grid(encoder, batch)
```

```{image} _static/encoder_token_grid.png
:alt: Encoded token value heatmap
:width: 100%
```

---

### plot_embedding_norms

Heatmap of per-token L2 embedding norms from the encoder output. Padding
positions are masked to white. Useful for a quick check that embedding
magnitudes are reasonable.

```python
from loomlib.viz import plot_embedding_norms

enc_output = encoder(batch)
fig = plot_embedding_norms(enc_output, batch)
```

```{image} _static/encoder_embedding_norms.png
:alt: Encoder embedding norms heatmap
:width: 100%
```

---

## Composing into custom layouts

Every function takes an optional `ax` keyword so you can embed plots in your
own figure grids:

```python
import matplotlib.pyplot as plt
from loomlib.viz import plot_allocation, plot_logits

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
plot_allocation(head, ax=ax1)
plot_logits(head, z, ax=ax2)
fig.tight_layout()
```

## Regenerating the figures in this page

The images above are generated by a script shipped with the docs:

```console
$ python docs/generate_figures.py
```
