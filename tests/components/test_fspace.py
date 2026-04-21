import shutil

import torch
import torch.nn.functional as F

from loomlib import FnSpace
from loomlib.thunker import VocabDecoder


fn_S = FnSpace()

fn_S.x = 1
fn_S.y = 2

fn_S.mult = lambda x, y : x.unsqueeze(1) * y.unsqueeze(0)
fn_S.add = lambda x, y : x.unsqueeze(1) + y.unsqueeze(0)

x = torch.randn(30)

print(fn_S(x, subFn=["mult"]))

space = FnSpace()

# Add types
space.x = 3 #VocabDecoder(choices=["a", "b", "c"], temp=1.0)
space.y = 3 #VocabDecoder(choices=["d", "e", "f"], temp=1.0)

# Add function
space.multiply = lambda x, y: x.unsqueeze(1) * y.unsqueeze(0)

def process(x, y):
    print("x: ", x)
    print("y: ", y)

space.print = process

# Evaluate
input_tensor = torch.randn(1, space.width)


print(space(input_tensor))  # Only returns 'multiply' output

fn_space = FnSpace()

fn_space.x = VocabDecoder(["cat", "dog", "fish"], temp=1.0)
space.y = VocabDecoder(choices=["d", "e", "f"], temp=1.0)

# Create a fake logit input
x = torch.randn(2, fn_space.width)

# Pretty print
fnspace = FnSpace()

my_thunker = VocabDecoder("abcd")
my_thunker2 = VocabDecoder("hijk")

fnspace.foo = 3                # identity thunk of width 3
fnspace.bar = (my_thunker, 4)  # custom thunk, explicit width
fnspace.baz = my_thunker2      # custom thunk with __len__ or .width
fnspace.myfunc = lambda foo: foo.mean()   # function
fnspace.print = lambda bar: print(bar)

x = torch.randn(2, fnspace.width)
print(fnspace(x))

print(fnspace.type_specs.keys())  # should show foo, bar, baz
print(fnspace.fns.keys())         # should show myfunc

#ls.pretty_print()
#out = ls(x)
#print(out)

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
