from lib import *
w = (7,6,5,4,3,2,1)

for k in range(2, len(w)-1):
    for R in consistent_sets(w, k):
        P = poset_to_hasse(admissible_poset(w, k, R))
        r = rank(P)
        if r is None:
            print("Counterexample")
            print(f"k = {k}")
            print(f"R = {R}")
            poset_to_graphviz(P)
