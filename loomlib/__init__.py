"""Loom: Weaving typed action spaces from transformer logits."""

__version__ = "0.1.0"

# Schema
from .schema import LoomModel, LoomUnion

# Type decoders
from .types import LoomType, Categorical, ContinuousScalar, BitInteger, Boolean, Scalar

# Compiler & Head
from .compiler import LoomCompiler
from .head import LoomHead

# Encoder
from .encoder import (
    LoomEncoder,
    LoomBatch,
    DefaultBranchEmbedding,
    XValEmbedding,
    ThermometerEmbedding,
    FourierValueEmbedding,
    GaussianBasisEmbedding,
)

# Layers (for advanced usage)
from .slice import SliceAllocation
from .fn import FnSpace
from .action import ActionDispatch

__all__ = [
    # Schema
    "LoomModel",
    "LoomUnion",
    # Types
    "LoomType",
    "Categorical",
    "ContinuousScalar",
    "BitInteger",
    "Boolean",
    "Scalar",
    # Compiler & Head
    "LoomCompiler",
    "LoomHead",
    # Encoder
    "LoomEncoder",
    "LoomBatch",
    "DefaultBranchEmbedding",
    "XValEmbedding",
    "ThermometerEmbedding",
    "FourierValueEmbedding",
    "GaussianBasisEmbedding",
    # Layers
    "SliceAllocation",
    "FnSpace",
    "ActionDispatch",
]
