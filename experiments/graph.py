from lib import *
w = (6,4,5,2,3,1)
R = frozenset([(1,2,5,6), (1,3,5,6)])
g = G(w, 4, R)
for x in g:
    print(f"{x}: ins = {g[x]['ins']}, outs = {g[x]['outs']}")