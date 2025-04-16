from lib import *
w = (2,5,4,3,1)
R = frozenset([(1,3,4,5)])
g = G(w, 4, R)
for x in g:
    print(f"{x}: ins = {g[x]['ins']}, outs = {g[x]['outs']}")