import torch
import torch.nn.functional as F
from loomlib import LogitSpace

# Basic Indexing

ls = LogitSpace()

ls.add("angle", slice(0, 3), lambda x: x.norm(dim=-1))
ls.add("target", [3, 4, 5],
       mean=lambda x: x.mean(-1),
       std=lambda x: x.std(-1),
       entropy=lambda x: -torch.sum(torch.softmax(x, -1) * torch.log_softmax(x, -1), -1))
ls.add("bonus", range(6, 9), lambda x: x.sum(-1), lambda x: x.prod(-1))

x = torch.randn(2, 9)
ls.pretty_print()
out = ls(x)
print(out)

#basic_toks = TokenChoice(["START", "STOP"])
#
#lspace.add(lambda x : basic_toks(F.softmax(x, dim=-1)), width=len(basic_toks))
#lspace.add(TokenChoice(data))
#
#print(lspace.names())
#
#x = torch.randn(10)
#
#y = lspace(x, lspace.names()[0])
#
#print(y)
#
#y = lspace(x)
#
#print(y)
#
