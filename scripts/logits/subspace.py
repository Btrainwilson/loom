import torch
from loomlib import LogitSpace

lspace = LogitSpace()
lspace[:10] = 


B = 2
N = 12


#lspace[5:9] = actionSpace

x = torch.randn((B,N))

clb = 5
cub = 10
pdist = torch.arange(cub - clb)

subspaces = [[0:5], [5:N]]

y = torch.mean(pdist * torch.functional.softmax(x[:, clb:cub]))
print(y)





#lspace = LogitSpace()
#
## Computes the expect
#
#pv = ActionSpace()
#
#pv.add_types({
#    "axis": list(range(1, 10)),
#    "angle": 
#})
#
#pv.add_action("SE", {"q1": "qubit", "q2": "qubit", "a" : "angle"}, handler=lambda q1, q2, a: print("SE:", q1, q2, a * 3.14 / 10))
#pv.add_action("DE", {"q1": "qubit", "q2": "qubit", "q3": "qubit", "q4": "qubit", "a" : "angle"}, handler=lambda q1, q2, q3, q4, a: print("DE:", q1, q2, q3, q4, a * 3.14 / 10))
#
#pv.compile()  # lock layout



