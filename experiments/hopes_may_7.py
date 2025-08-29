from lib import *
w = (14, -14, -7, 20, 2)
# Maximal element in congruence poset is 5.
# Minimal element is 1.

# w |- 1 = [2, 16, -7, -1]
# h = 0
# R = {(1, 7, 19, 20), (3, 7, 19, 20), (1, 3, 19, 20), (1, 3, 24, 25),  (1, 3, 7, 9),  (1, 2, 14, 15), (1, 2, 9, 10), (1, 2, 4, 5)}
# R|h  {(1,2,4,5)}
# fl(R|h) = {124}
# Inv_k|h = {125, 145, 245, 135, 345}
# R'_h =Inv_k-1(w') setminus fl(R|h) = {(1, 3, 12), (1, 6, 16), (1, 10, 20), (1, 3, 8), (3, 6, 16), (1, 6, 12), (1, 10, 16), (1, 3, 4), (1, 3, 20), (1, 2, 12), (1, 3, 16), (3, 6, 12), (1, 6, 8), (1, 10, 12), (3, 6, 8), (1, 3, 6), (1, 2, 8)}
# G_{R'_h} has 19 vertices, but Inv_k|h has size 5?

def iX(X):
    k = len(X)
    n = len(w)
    i = 0
    while i*n < X[-1]:
        i += 1
    while X + tuple([i*n]) in inv(w, k+1):
        i += 1
    return i

# Need some help understanding the indices for Hope 3, got confused by k-1, k, k+1.

print(inv(w,5))
for X in inv(w, 3):
    if X[-1]%5 != 0:
        print(X, iX(X))