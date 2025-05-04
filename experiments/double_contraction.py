import itertools
from lib import *

for w in itertools.permutations([1,2,3,4,5,6]):
    n = len(w)
    for R in consistent_sets(w, n-1):
        G_R = G(w, n-1, R)
        for X in G_R:
            for Y in G_R[X]["outs"]:
                if X == Y:
                    continue
                if n in X or n in Y:
                    continue
                for j in range(1, n-1):
                    if (j in X and j in Y) and (j+1 in X and j+1 in Y):
                        v = F(F(w, j+1), j)
                        C = consistent_sets(v, n-3)
                        cR = frozenset([contraction(contraction(X,j+1),j) for X in R if (j in X and j+1 in X)])
                        if cR not in C:
                            print(f"Counterexample w = {w}, j = {j}, R = {R}, cR = {cR}, C  = {C}, X = {X}, Y = {Y}")
