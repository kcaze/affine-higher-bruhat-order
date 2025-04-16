from typing import Generic, Iterator, TypeVar, TypedDict, Union
import copy, functools, itertools, sys, math
from functools import lru_cache

T = TypeVar('T')
Perm = tuple[int, ...]
Inv = tuple[int, ...]
class PosetEntry(TypedDict):
    ins: set
    outs: set
Poset = dict[T, PosetEntry]

def repr(x: int, n: int) -> int:
  """Returns the congruence representative of x in the range [1, n]"""
  return n if x%n == 0 else x%n

def inverse(w: Perm) -> Perm:
  """Returns the inverse permutation w^-1"""
  n: int = len(w)
  w_inv = [0 for _ in range(n)]
  for i in range(n):
    j = repr(w[i], n)
    w_inv[j-1] = i + (j - w[i])
  return tuple(w_inv)

@lru_cache
def inv(w: Perm, k: int) -> set[Inv]:
    """Returns all k-inversions of a permutation w."""
    ret: set[Inv] = set()
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

def quasi_inv(w: Perm, k: int) -> set[Inv]:
    n = len(w)
    inv_k_1 = inv(w, k-1)
    ret = set()
    for X in itertools.combinations(range(1, n+1), k):
        P_X = set(packet(X))
        if len(P_X & inv_k_1) == 2 and k-1 > 2:
            ret.add(X)
    return ret

@lru_cache
def inv_contraction(w: Perm, k: int) -> set[Inv]:
    n = len(w)
    return set([X for X in inv(w,k) if X[-1]%n==0])

def contraction(x: Inv, j: int) -> Inv:
    x = [(i if i < j else i-1) for i in x if i != j]
    return tuple(x)

def deletion(x: Inv, j: int) -> Inv:
    x = [(i if i < j else i-1) for i in x if i != j]
    return tuple(x)

def truncation(w: Perm) -> Perm:
    n = len(w)
    m = w.index(n)+1
    y = w[m:]
    sy = sorted(y)
    return tuple([sy.index(i)+1 for i in y])

def truncation_inv(w: Perm, x: Inv) -> Inv:
    n = len(w)
    assert(x[-1] == n)
    m = w.index(n)+1
    tw = truncation(w)
    return tuple([tw[w.index(i) - m] for i in x[:-1]])

def F(w: Perm, j:int) -> Perm:
    return tuple([i if i < j else i-1 for i in w if i != j])

def G(w: Perm, k: int, R: frozenset[Inv]) -> Poset[Inv]:
    V = inv(w, k-1)
    ret: Poset[Inv] = {}
    for x in V:
        ret[x] = {'ins':set(), 'outs':set()}
    for X in inv(w, k):
        P = packet(X)
        lex = X not in R
        for i in range(len(P)-1):
            if not lex:
                ret[P[i]]['outs'].add(P[i+1])
                ret[P[i+1]]['ins'].add(P[i])
            else:
                ret[P[i]]['ins'].add(P[i+1])
                ret[P[i+1]]['outs'].add(P[i])
    for X in quasi_inv(w, k):
        P = packet(X)
        for i in range(len(P)-1):
            if P[i] in V:
                break
        print(i, k, P, P[i], P[i+1])
        if (k-i) % 2 == 1:
            ret[P[i]]['outs'].add(P[i+1])
            ret[P[i+1]]['ins'].add(P[i])
        else:
            ret[P[i]]['ins'].add(P[i+1])
            ret[P[i+1]]['outs'].add(P[i])
    return ret


def normalize(x: Inv, n: int) -> Inv:
    shift = repr(x[0],n) - x[0]
    return tuple(i+shift for i in x)

def packet(x: Inv) -> list[Inv]:
    """Returns the packet of x in antilex order."""
    ret: list[Inv] = []
    for i in range(len(x)):
        ret.append(tuple(x[:i] + x[(i+1):]))
    return ret

def affine_shift(x: Inv, n: int) -> tuple[int]:
    """Returns the shift in the v-basis."""
    k = len(x)
    ret = []
    for i in range(k-1):
        ret.append((x[i+1]-x[i])//n)
    return tuple(ret)[::-1]

def affine_shift_join(shift1: tuple[int], shift2: tuple[int]) -> tuple[int]:
    """Returns the join of 2 shifts in the v-basis."""
    return [(min if i%2 == 1 else max)(shift1[i], shift2[i]) for i in range(len(shift1))]

def permutations(n: int, l: int, affine: bool = True, n_maximal: bool = True) -> set[Perm]:
  """ Returns all permutations in affine S_n of length l."""
  ret = set()
  if l == 0:
    ret.add(tuple(range(1, n+1)))
  else:
    for w in permutations(n, l-1):
      for i in range(n) if affine else range(n-1):
        if i != n-1 and w[i] < w[i+1] and (w[i] % n != 0 if n_maximal else True):
          v = list(w)
          v[i], v[i+1] = v[i+1], v[i]
          ret.add(tuple(v))
        elif i == n-1 and w[-1] < w[0] + n and (w[-1]%n != 0 if n_maximal else True):
          v = list(w)
          v[0], v[-1] = v[-1]-n, v[0]+n
          ret.add(tuple(v))
  return ret

def is_congruent(i: Inv, j: Inv, n: int) -> bool:
    """Return if i == j (mod n)"""
    return all(i[k] % n == j[k] % n for k in range(len(i)))

def common_packet(i: Inv, j: Inv, n: int) -> Union[Inv, None]:
    """Returns X such that i,j in P(X). Returns None if no such X exists."""
    k = len(i)
    c = set(repr(x,n) for x in i) | set(repr(x,n) for x in j)

    if len(c) != k+1:
        return
    
    shift = None
    if repr(i[0],n) == repr(j[0],n):
        shift = i[0] - j[0]
    elif repr(i[-1],n) == repr(j[-1],n):
        shift = i[-1] - j[-1]
    elif repr(i[0],n) == repr(j[1],n):
        shift = i[0] - j[1]
    elif repr(i[1], n) == repr(j[0], n):
        shift = i[1] - j[0]
    if shift is None:
        return

    j = tuple(x+shift for x in j)
    ret = sorted(list(set(i) | set(j)))
    if len(ret) != k+1:
        return
    shift = repr(ret[0],n) - ret[0]

    return tuple(x+shift for x in ret)

def permanent_poset(w: Perm, k: int) -> Poset[Inv]:
    """Permanent poset on Inv_k(w)"""
    ret = {}
    n = len(w)
    I = inv(w,k)
    J = inv(w,k+1)
    for i in I:
        ret[i] = {"ins": set(), "outs": set()}

    for i in I:
        for j in I:
            if i == j:
                continue
            if is_congruent(i, j, n):
                shift = [j[idx] - i[idx] for idx in range(k)][::-1]
                idx = 0 
                while shift[idx] == n:
                    idx += 1
                if not all(x == 0 for x in shift[idx:]):
                    continue
                if idx%2 == 0:
                    ret[j]["outs"].add(i)
                    ret[i]["ins"].add(j)
                else:
                    ret[j]["ins"].add(i)
                    ret[i]["outs"].add(j)
            else:
                X = common_packet(i,j,n)
                if X and X not in J:
                    P = packet(X,n)[::-1]
                    if i not in P:
                        continue
                    i_idx = P.index(i)
                    j_idx = P.index(j)
                    if i_idx > j_idx:
                        continue
                    if j_idx - i_idx > 1:
                        print("BIG ERROR")
                    if i_idx%2 == 1:
                        ret[j]["outs"].add(i)
                        ret[i]["ins"].add(j)
                    else:
                        ret[j]["ins"].add(i)
                        ret[i]["outs"].add(j)
    return ret

def packets_containing(w: Perm, x: Inv) -> list[Inv]:
    """Returns all full packets y such that x in P(y)."""
    n = len(w)
    k = len(x)
    ret: list[Inv] = []
    for y in inv(w, k+1):
        if x in packet(y, n):
            ret.append(y)
    return ret

def is_consistent_on(w: Perm, k: int, rev: frozenset[Inv], xs: list[Inv]) -> bool:
    """For x in xs, check if reversal set rev is consistent w.r.t. x.
    rev should be a subset of inv(w, k).
    Returns True if consistent w.r.t. all x in xs and False otherwise."""

    n = len(w)
    for x in xs:
        p = packet(x,n)
        # Rev is either prefix or suffix of p.
        flip = []
        for i in range(k):
            if (p[i] in rev) != (p[i+1] in rev):
                flip.append(i)
        if len(flip) > 1:
            return False
    return True

def consistent_sets(w: Perm, k: int) -> set[frozenset[Inv]]:
    """Return all consistent reversal subsets of inv(w, k)"""
    n: int = len(w)
    ret: set[frozenset[Inv]] = set([frozenset()])
    todo: list[frozenset[Inv]] = [frozenset()]
    P = permanent_poset(w, k)

    while len(todo) > 0:
        curr = todo.pop()
        for iv in inv(w,k):
            satisfies_P = True
            if iv in P:
                for jv in P[iv]["ins"]:
                    if jv not in curr:
                        satisfies_P = False
                        break
            if not satisfies_P:
                continue

            new = frozenset(curr) | frozenset([iv])
            if new not in ret and is_consistent_on(w, k, new, packets_containing(w, iv)):
                ret.add(new)
                todo.append(new)
    return ret

def admissible_poset(w: Perm, k: int, R: frozenset[Inv]) -> Poset[Inv]:
    n = len(w)
    ret = permanent_poset(w, k)
    for x in inv(w, k):
        if x not in ret:
            ret[x] = {"ins": set(), "outs": set()}
    for x in inv(w,k+1):
        P = packet(x, n)
        if x in R:
            P = P[::-1]
        for i in range(k):
            for j in range(i+1, k+1):
                ret[P[j]]["outs"].add(P[i])
                ret[P[i]]["ins"].add(P[j])
    return ret

def congruence_poset(w: Perm) -> Poset[int]:
  n = len(w)
  ret = {}
  for i in range(1, n+1):
    ret[i] = {"ins": set(), "outs": set()}
  for (i,j) in inv(w,2):
    ret[repr(i,n)]["outs"].add(repr(j,n))
    ret[repr(j,n)]["ins"].add(repr(i,n))
  return ret

def lower_order_ideals(P: Poset) -> set:
    ret = set()
    P = transitive_closure(P)
    all_elems = set(P.keys())
    none_elems = frozenset()
    ret.add(none_elems)
    todo = [none_elems]
    while len(todo) > 0:
        E = todo.pop()
        for X in all_elems:
            if X in E:
                continue
            if not (P[X]["ins"] <= E):
                continue
            F = set(E)
            F.add(X)
            F = frozenset(F)
            if F not in ret:
                ret.add(F)
                todo.append(F)
    return ret

def rotate_consistent_set(w: Perm, k: int, rev: frozenset[Inv]) -> frozenset[Inv]:
    n = len(w)
    new = set()
    for iv in inv(w,k):
        if 1 in iv:
            if iv not in rev:
                vi = tuple(i-1 for i in iv[1:]) + tuple([n])
                new.add(vi)
        elif iv in rev:
            new.add(tuple(i-1 for i in iv))
    return frozenset(new)

def transitive_closure(P: Poset) -> Poset:
    ret: Poset = {x: {"ins": set(), "outs": set()} for x in P}
    for X in P:
        ret[X]["ins"] = set(P[X]["ins"])
        ret[X]["outs"] = set(P[X]["outs"])
    for i in range(len(P)):
        changed = False
        for X in ret:
            outs = list(ret[X]["outs"])
            for Y in outs:
                for Z in ret[Y]["outs"]:
                    if Z not in outs:
                        changed = True
                        ret[X]["outs"].add(Z)
                        ret[Z]["ins"].add(X)
        if not changed:
            break
    return ret

def poset_to_hasse(P: Poset) -> Poset:
    ret: Poset = {x: {"ins": set(), "outs": set()} for x in P}
    P = transitive_closure(P)
    for x in P:
        for y in P[x]["outs"]:
            if len(P[x]["outs"] & P[y]["ins"]) == 0:
                ret[x]["outs"].add(y)
                ret[y]["ins"].add(x)
    return ret

def shift(w: Inv) -> Inv:
    """Return the n-shift of a permutation."""
    n = len(w)
    for i in range(n):
        if w[i]%n == 0:
            break
    v = list(w)
    v[i] += n
    v = [v[-1]-n] + v[:-1] 
    return tuple(v)

def unshift(w: Inv) -> Inv:
    """Return the n-unshift of a permutation."""
    n = len(w)
    for i in range(n):
        if w[i]%n == 0:
            break
    v = list(w)
    v[i] -= n
    v = v[1:] + [v[0]+n] 
    return tuple(v)

def contraction_stratification(w: Inv, k: int):
    n = len(w)
    ws = [w]
    while len(inv_contraction(ws[-1],k)) > 0:
        ws.append(unshift(ws[-1]))
    ws = ws[::-1]
    layers = [[set()]]
    for i in range(len(ws)-1):
        existing = set()
        for L in layers:
            for X in L:
                existing = existing | X
        curr_layer = [inv_contraction(ws[i+1],k)-existing]
        for _ in range(len(ws)-i-2):
            I = []
            for X in curr_layer[-1]:
                Y = list(X)
                Y[-1] += n
                I.append(tuple(Y))
            curr_layer.append(set(I))
        layers.append(curr_layer)
    layers = layers[1:][::-1]
    return layers

def stratification_to_labels(w: Inv, k: int, layers):
    inv_to_tuple = {}
    for X in inv_contraction(w,k):
        for i in range(len(layers)):
            for j in range(len(layers[i])):
                if X in layers[i][j]:
                    inv_to_tuple[X] = f"{X};({i},{j})"
    return inv_to_tuple

def poset_to_graphviz(P: Poset, n=None, hasse=False, stratification=None, full_edges=None, restriction_set=None, w=None):
    print("digraph G {")
    print("rankdir=BT;")

    depths = {}
    heights = {}
    for si in P:
        if si[-1]%n == 0:
            # heights[si] = (si[-1] - si[-2])//n
            # heights[si] = coheight(si, w)
            heights[si] = (si[-1] - si[-2])//n - coheight(si, w)
            depths[si] = set()
            for sj in P:
                if sj[:-1] == si[:-1] and sj[-1]%n == 0:
                    depths[si].add(sj)
            depths[si] = len(depths[si])-1
    max_depth = max(depths.values())
    min_height = min(heights.values())
    if min_height < 0:
        for x in heights:
            heights[x] += abs(min_height)

    if restriction_set != None:
        P = restriction(P, restriction_set)
    if hasse:
        P = poset_to_hasse(P)


    for si in P:
        if n:
            if si[-1]%n != 0:
                print(f'"{si}" [color=lightgray, fontcolor=lightgray];')
            else:
                colors = ["red", "orange", "green", "blue", "purple"]
                # colors = ["#E6194B", "#F58231", "#3CB44B", "#4363D8", "#911EB4"]
                # fillcolors = ["#FABED4", "#FFD8B1", "#AAFFC3", "#BDC8F1", "#DCBEFF"]
                fillcolors = ["#FFFFFF", "#DDDDDD", "#BBBBBB", "#999999", "#777777"]

                # Color based on last gap.
                # c = colors[(si[-1] - si[-2])//n]
            
                # Color based on recency added with n-shifts
                # congruent_sis = sorted([X for X in P if X[:-1] == si[:-1] and X[-1]%n == 0])[::-1]
                # c = colors[max_gap - 1 - congruent_sis.index(si)]
                c = colors[heights[si]]
                fc = fillcolors[max_depth - depths[si]]
                if stratification:
                    print(f'"{stratification[si]}" [color="{c}", fillcolor="{fc}", style="filled", penwidth=5];')
                else:
                    print(f'"{si}" [color="{c}", fillcolor="{fc}", style="filled", penwidth=5];')
        else:
            print(f'"{si}";')
        for sj in P[si]["outs"]:
            is_full = full_edges and frozenset([si, sj]) in full_edges

            if n:
                if si[-1]%n != 0 or sj[-1]%n != 0:
                    if stratification:
                        print(f'"{stratification[si] if si in stratification else si}" -> "{stratification[sj] if sj in stratification else sj}" [color=lightgray];')
                    else:
                        print(f'"{si}" -> "{sj}" [color=lightgray];')

                else:
                    if stratification:
                        print(f'"{stratification[si]}" -> "{stratification[sj]}" [color={"black" if is_full else "red"}];')
                    else:
                        print(f'"{si}" -> "{sj}" [color={"black" if is_full else "red"}];')
            else:
                print(f'"{si}" -> "{sj}" [color={"black" if is_full else "red"}];')
    print("}")

def is_connected(P: Poset) -> bool:
    if len(P) == 0:
        return True
    X = list(P.keys())[0]
    ret = set([X])
    todo = [X]
    while len(todo) > 0:
        X = todo.pop()
        for Y in (P[X]["outs"] | P[X]["ins"]):
            if Y not in ret:
                ret.add(Y)
                todo.append(Y)
    return len(ret) == len(P)

def restriction(P: Poset, elems) -> Poset:
    Q = {x: {"ins": set(), "outs": set()} for x in elems}
    for x in Q:
        for y in P[x]["ins"] & elems:
            Q[x]["ins"].add(y)
        for y in P[x]["outs"] & elems:
            Q[x]["outs"].add(y)
    return Q

def permutation_contraction(w: Perm, n: int) -> Perm:
    v = [x - x//n for x in w if x%n != 0]
    while sum(v) < n*(n-1)//2:
        v = v[1:] + [v[0]+(n-1)]
    while sum(v) > n*(n-1)//2:
        v = [v[-1]-(n-1)] + v[:-1]
    return v

@lru_cache(maxsize=4096)
def depth(X: Inv, w: Perm) -> int:
    n = len(w)
    k = len(X)
    invs = inv(w, k)
    X = list(X)
    while tuple(X) in invs:
        X[-1] += n
    return (X[-1] - X[-2])//n

@lru_cache(maxsize=4096)
def height(X: Inv, w: Perm) -> int:
    return (X[-1] - X[-2])//len(w)+1

@lru_cache(maxsize=4096)
def coheight(X: Inv, w: Perm) -> int:
    n = len(w)
    for i in range(n):
        if w[i]%n == X[-1]%n:
            break
    for j in range(n):
        if w[j]%n == X[-2]%n:
            break
    i = i + X[-1] - w[i]
    j = j + X[-2] - w[j]
    return (j-i)//len(w)

def rank(P: Poset):
    rank = {}
    todo = set()
    for k in P:
        if len(P[k]["ins"]) == 0:
            todo.add(k)
    r = 0
    while len(todo) > 0:
        for k in todo:
            if k in rank:
                return None
            rank[k] = r
        new_todo = set()
        for k in todo:
            for k_ in P[k]["outs"]:
                new_todo.add(k_)
        todo = new_todo
        r += 1
    return rank