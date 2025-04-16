from lib import *
w = (6,4,5,3,2,1)

for j in range(1, len(w)+1):
    v = contraction(w, j)
    for k in range(2, len(w)-1):
        C = consistent_sets(v, k)
        for R in consistent_sets(w, k):
            S = set([contraction(X,j) for X in R if j not in X])
            print(R,S,j)
            if S not in C:
                print(f"Counterexample w = {w}, j = {j}, k = {k}, R = {R}")