from loomlib import ActionSpace

pv = ActionSpace()

pv.add_types({
    "direction": ["N", "S", "E", "W"],
    "force": list(range(1, 20)),
    "select": list(range(1, 20)),
})

pv.add_action("Move", {"dir": "direction", "q2": "qubit", "a" : "angle"}, handler=lambda q1, q2, a: print("SE:", q1, q2, a * 3.14 / 10))
pv.add_action("DE", {"q1": "qubit", "q2": "qubit", "q3": "qubit", "q4": "qubit", "a" : "angle"}, handler=lambda q1, q2, q3, q4, a: print("DE:", q1, q2, q3, q4, a * 3.14 / 10))

tokens = pv.encode([
    {"SE": {"q1": 2, "q2": 6, "a": 1 }},
    {"SE": {"q1": 3, "q2": 8, "a": 2 }},
    {"DE": {"q1": 1, "q2": 4, "q3": 6, "q4": 9, "a": 5 }},
])
print("Tokens:", tokens)

decoded = pv.decode(tokens)
print("Decoded:", decoded)

pv.run(decoded)
pv.print_vocab()

pv.print_run(decoded)
