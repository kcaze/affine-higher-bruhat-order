from lib import *
import itertools

for w in itertools.permutations([1,2,3,4,5,6,7]):
    v = truncation(w)
    n = len(w)
        
    for k in range(3, len(w)):
        tC = consistent_sets(v, k)
        for R in consistent_sets(w, k+1):
            tR = set([truncation_inv(w, X) for X in R if X[-1] == n])
            S = frozenset([X for X in inv(v, k) if X not in tR])
            if S not in tC:
                print(f"Counterexample w = {w}, wTn = {v}, k = {k}, R = {R}, RTn = {tR}")