"""Canonical Loom color palette.

These hex values mirror the CSS custom-property palette defined in
``docs/_static/custom.css`` (the ``--loom-*`` variables).  Every
visualization module in ``loomlib.viz`` imports colors from here so
that figures stay consistent with the documentation theme.
"""

BLUE: str = "#3B3E63"
CYAN: str = "#EB6254"
GREEN: str = "#5E5D4C"
YELLOW: str = "#D9BC65"
RED: str = "#E58D58"
PURPLE: str = "#3B3E63"
GREY: str = "#3E3E3E"
BLACK: str = "#7A7A7A"

PALETTE: list[str] = [BLUE, CYAN, GREEN, YELLOW, RED, GREY, BLACK]
"""Ordered cycle used for branch / category coloring."""

TYPE_COLORS: dict[str, str] = {
    "Categorical": BLUE,
    "ContinuousScalar": GREEN,
    "BitInteger": YELLOW,
    "Boolean": CYAN,
    "Scalar": PURPLE,
}
DEFAULT_COLOR: str = GREY

OPCODE_COLOR: str = BLACK
PADDING_COLOR: str = "#C0C0C0"
