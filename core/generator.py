from .types import *

def permutations(n: int, l: int, affine: bool = True) -> set[perm]:
  """ Returns all permutations in (affine) S_n of length l."""
  ret = set()
  if l == 0:
    ret.add(tuple(range(1, n+1)))
  else:
    for w in permutations(n, l-1, affine):
      for i in range(n) if affine else range(n-1):
        if i != n-1 and w[i] < w[i+1]:
          v = list(w)
          v[i], v[i+1] = v[i+1], v[i]
          ret.add(tuple(v))
        elif i == n-1 and w[-1] < w[0] + n:
          v = list(w)
          v[0], v[-1] = v[-1]-n, v[0]+n
          ret.add(tuple(v))
  return ret
