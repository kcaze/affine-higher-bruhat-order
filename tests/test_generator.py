from ..core.generator import *

def test_permutations_nonaffine():
    assert permutations(3, 0, False) == {(1,2,3)}
    assert permutations(3, 1, False) == {
        (2,1,3),
        (1,3,2),
    }
    assert permutations(3, 2, False) == {
        (2,3,1),
        (3,1,2),
    }
    assert permutations(3, 3, False) == {
        (3,2,1),
    }
    assert permutations(3, 4, False) == set()
    assert permutations(3, 5, False) == set()

def test_permutations_affine():
    assert permutations(3, 0) == {(1,2,3)}
    assert permutations(3, 1) == {
        (2,1,3),
        (1,3,2),
        (0,2,4),
    }
    assert permutations(3, 2) == {
        (2,3,1),
        (0,1,5),
        (3,1,2),
        (-1,3,4),
        (0,4,2),
        (2,0,4),
    }
    assert permutations(3, 3) == {
        (3,2,1),
        (-2,3,5),
        (1,0,5),
        (0,5,1),
        (-1,1,6),
        (3,-1,4),
        (-1,4,3),
        (4,0,2),
        (-1,4,3),
        (2,4,0),
        (1,0,5)
    }