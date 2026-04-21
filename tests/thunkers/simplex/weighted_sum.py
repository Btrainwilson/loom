from loomlib.thunker.simplex import greedy_fill_left
import torch
from tensor_mosaic import Mosaic
from loomlib.thunker.types import WeightedSum


def test_greedy_fill_left_single():
    mass = torch.tensor([4., 2., 6.])
    value = 7.0
    expected = torch.tensor([4., 2., 1.])
    alloc, p = greedy_fill_left(mass, value)
    assert torch.allclose(alloc, expected)
    assert torch.allclose(p, expected / mass)

def test_greedy_fill_left_batch():
    mass = torch.tensor([[4., 2., 6.], [3., 1., 5.]])
    value = torch.tensor([7.0, 6.0])
    expected = torch.tensor([[4., 2., 1.],
                             [3., 1., 2.]])
    alloc, p = greedy_fill_left(mass, value)
    assert torch.allclose(alloc, expected)
    assert torch.allclose(p, expected / mass)

def test_weightedsum_fill_left_scalar():
    mspace = Mosaic()
    mspace.DUMMY = 3
    mass = torch.tensor([4., 2., 6.])
    value = 7.0
    ws = WeightedSum(mass, encoding_strategy="fill_left", subspace=mspace.DUMMY)
    out = ws.encode(value)
    expected = torch.tensor([4., 2., 1.])
    assert torch.allclose(out, expected)

def test_weightedsum_fill_left_batch():
    mspace = Mosaic(cache=False)
    mspace.DUMMY = (2, 3)
    print("DUMMY: ", mspace.DUMMY)
    mass = torch.tensor([[4., 2., 6.], [3., 1., 5.]])
    value = torch.tensor([7.0, 6.0])
    ws = WeightedSum(mass, encoding_strategy="fill_left", subspace=mspace.DUMMY)
    out = ws.encode(value)
    expected = torch.tensor([[4., 2., 1.],
                             [3., 1., 2.]])
    assert torch.allclose(out, expected)

def test_weightedsum_proportional_scalar():
    mspace = Mosaic()
    mspace.DUMMY = 3
    mass = torch.tensor([1., 2., 1.])
    value = 4.0
    ws = WeightedSum(mass, encoding_strategy="fill_left", subspace=mspace.DUMMY)
    out = ws.encode(value)
    # Proportional split: (1,2,1) sum=4 -> [1,2,1]
    expected = torch.tensor([1., 2., 1.])
    assert torch.allclose(out, expected)

def test_weightedsum_fill_left_mass_smaller_than_value():
    mspace = Mosaic()
    mspace.DUMMY = 3
    mass = torch.tensor([1., 2., 1.])
    value = 10.0
    ws = WeightedSum(mass, encoding_strategy="fill_left", subspace=mspace.DUMMY)
    out = ws.encode(value)
    # Should allocate all mass (since sum < value)
    expected = mass
    assert torch.allclose(out, expected)

def test_weightedsum_fill_left_zero_mass():
    mspace = Mosaic()
    mspace.DUMMY = 3
    mass = torch.tensor([0., 0., 0.])
    value = 5.0
    ws = WeightedSum(mass, encoding_strategy="fill_left", subspace=mspace.DUMMY)
    out = ws.encode(value)
    expected = torch.tensor([0., 0., 0.])
    assert torch.allclose(out, expected)

# Run all tests if executed directly
if __name__ == "__main__":
    test_greedy_fill_left_single()
    test_greedy_fill_left_batch()
    test_weightedsum_fill_left_scalar()
    test_weightedsum_fill_left_batch()
    test_weightedsum_proportional_scalar()
    test_weightedsum_fill_left_mass_smaller_than_value()
    test_weightedsum_fill_left_zero_mass()
    print("All tests passed!")
