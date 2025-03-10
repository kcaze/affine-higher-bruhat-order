from functools import lru_cache
import itertools

# Types
perm = tuple[int]
inv = tuple[int]

def cong_repr(x: int, n: int) -> int:
  """Returns the congruence representative of x in the range [1, n]"""
  return n if x%n == 0 else x%n

def inverse(w: perm) -> perm:
  """Returns the inverse permutation w^-1"""
  n: int = len(w)
  w_inv = [0 for _ in range(n)]
  for i in range(n):
    j = cong_repr(w[i], n)
    w_inv[j-1] = (i+1) + (j - w[i])
  return tuple(w_inv)

@lru_cache
def inversions(w: perm, k: int) -> set[inv]:
    """Returns all k-inversions of a permutation w."""
    ret: set[inv] = set()
    n: int = len(w)
    w_inv = inverse(w)

    def inv_helper(idxs: list[int], vals: list[int]) -> list[list[int]]:
        if len(idxs) == 1:
            return [[idxs[0]]]

        ret = []
        i = 0
        while vals[0] - vals[1] > i*n:
            new_idxs = [x + i*n for x in idxs[1:]]
            new_vals = [x + i*n for x in vals[1:]]
            for r in inv_helper(new_idxs, new_vals):
                ret.append([idxs[0]] + r) 
            i += 1
        return ret

    for xs in itertools.permutations(range(1, n+1), k):
        idxs = list(xs)
        vals = [w_inv[i-1] for i in idxs]

        for i in range(1, k):
            while idxs[i] < idxs[i-1]:
                idxs[i] += n
                vals[i] += n
        for r in inv_helper(idxs, vals):
            ret.add(tuple(r))
    return ret