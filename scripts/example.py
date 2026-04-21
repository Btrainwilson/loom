# example_usage.py
from loomlib import ActionSpace

aspace = ActionSpace()

K = 10
types = {
    "theta": [l * 3.14 * 2 / K for l in range(1, K)],
    "dist": [l * 0.1 for l in range(1, K)],
    "compass": ["NORTH", "SOUTH", "EAST", "WEST"],
}

aspace.types(types)

aspace.add(
    {
        "Move": {"x": "dist", "y": "dist", "fn": lambda x, y: print((x, y))  }
        "Rotate": {"a": "angles", "d": ["CLOCKWISE", "ANTICLOCKWISE"], "fn": lambda a, d : print(a, d) }
    }
)



pv = ParamVocabulary()

pv.add_types({
    "dist": [0.1, 0.2],
    "angle": [90, 180],
    "dir": ["CW", "CCW"]
})

pv.add_action("Move", {"x": "dist", "y": "dist"}, handler=lambda x, y: print("Move", x, y))
pv.add_action("Rotate", {"a": "angle", "d": "dir"}, handler=lambda a, d: print("Rotate", a, d))


pv.add_action("Norm", {"x": "dist", "y": "dist"}, handler=lambda x, y: print("Norm", (x**2 + y**2)**.5))
pv.add_action("Rotate2EB", {"a": "angle", "d": "dir"}, handler=lambda a, d: print("Rotate", a, d))

pv.compile()  # lock layout

tokens = pv.encode([
    {"Norm": {"x": 0.1, "y": 0.2}},
    {"Rotate": {"a": 180, "d": "CCW"}},
])
print("Tokens:", tokens)

decoded = pv.decode(tokens)
print("Decoded:", decoded)

pv.run(decoded)
pv.print_vocab()
