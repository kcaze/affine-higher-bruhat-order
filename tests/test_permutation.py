import pytest
from ..core.permutation import *

def test_cong_repr():
    assert cong_repr(1, 5) == 1
    assert cong_repr(21, 5) == 1
    assert cong_repr(-4, 5) == 1
    assert cong_repr(53, 5) == 3
    assert cong_repr(-5, 5) == 5
    assert cong_repr(10, 5) == 5

def test_inverse():
    assert inverse((1,2,3,4,5)) == (1,2,3,4,5)
    assert inverse((5,4,3,2,1)) == (5,4,3,2,1)
    assert inverse((2,3,1)) == (3,1,2)
    assert inverse((-3, -2, 8, 7)) == (5, 6, 0, -1)

def test_inversions():
    # All numbers are 1-inversions.
    assert inversions((1,2,3,4,5), 1) == {(1,),(2,),(3,),(4,),(5,)}

    for k in range(2, 6):
        assert inversions((1,2,3,4,5), k) == set()
    
    assert inversions((-3, -2, 8, 7), 2) == {(1,3), (1,4), (1,7), (1,8), (2,3), (2,4), (2,7), (2,8), (3,4)}
    assert inversions((-3, -2, 8, 7), 3) == {(1,3,4), (1,7,8), (2,3,4), (2,7,8)}
    assert inversions((-3, -2, 8, 7), 4) == set()

def test_inversion_multigraph():
    assert(inversion_multigraph((-11, -2, 15, 11, 2))) == (
        (0,1,5,4,2),
        (-1,0,3,2,0),
        (-5,-3,0,-1,-3),
        (-4,-2,1,0,-2),
        (-2,0,3,2,0),
    )