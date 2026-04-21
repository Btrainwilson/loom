import torch
import torch.nn.functional as F

from loomlib import LogitSpace, TokenChoice, ActionSpace
from loomlib.aspace import ActionSpace, Action, AType, AParam

xStrs = ["HI", "YE", "PHI"]
angleType = AReal(0.0, 3.14, 10)

aspace = ActionSpace()

aspace += Action("TEST1",
                 x=xStrs,
                 y=angleType,
                 action_fn = lambda x, y : print(x, y)
                 )



lspace = LogitSpace()


basic_toks = TokenChoice(["START", "STOP"])

lspace.add(lambda x : basic_toks(F.softmax(x, dim=-1)), width=len(basic_toks))
lspace.add(TokenChoice(data))

print(lspace.names())

x = torch.randn(10)

y = lspace(x, lspace.names()[0])

print(y)

y = lspace(x)

print(y)

