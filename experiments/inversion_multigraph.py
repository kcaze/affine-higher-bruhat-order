from core.generator import *
from core.permutation import *
from core.types import *
import itertools
from tqdm import tqdm

def multigraphs_satisfying_triangle_inequality(n, max_weight):
    for wts in itertools.product(range(-max_weight, max_weight+1), repeat=n*(n-1)//2):
        G = [[0] * n for _ in range(n)]
        idx = 0
        for i in range(n):
            for j in range(i+1, n):
                weight = wts[idx]
                G[i][j] = weight
                G[j][i] = -weight
                idx += 1
        is_valid = True
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    ik = 1 if G[i][k] >= 0 else -1
                    ij = 1 if G[i][j] >= 0 else -1
                    jk = 1 if G[j][k] >= 0 else -1
                    if (ik,ij,jk) in [(-1,1,1), (1,-1,-1)]:
                        is_valid = False

                    A = i if G[i][j] >= 0 and G[i][k] >= 0 else j if G[j][i] >= 0 and G[j][k] >= 0 else k
                    C = k if G[j][k] >= 0 and G[i][k] >= 0 else j if G[i][j] >= 0 and G[k][j] >= 0 else i
                    B = [x for x in [i,j,k] if x != A and x != C][0]
                    if (A,B,C) in [(i,j,k), (k,i,j), (j,k,i)]:
                        if G[A][B] - G[A][C] + G[B][C] not in [0, -1]:
                            is_valid = False
                    else:
                        if G[A][B] - G[A][C] + G[B][C] not in [0, 1]:
                            is_valid = False
        if is_valid:
            yield tuple(tuple(x) for x in G)

def experiment_difference_values():
    difference_values = set()
    for n in tqdm(range(3, 10)):
        for length in tqdm(range(1, 10), leave=False):
            for w in tqdm(permutations(n, length), leave=False):
                G = inversion_multigraph(w)
                for a, b, c in itertools.combinations(range(n), 3):
                    A = min([a,b,c], key = lambda i: w[i])
                    C = max([a,b,c], key = lambda i: w[i])
                    B = [x for x in [a,b,c] if x != A and x != C][0]

                    # A --> B
                    #   \  |
                    #    \ |
                    #      v
                    #      C
                    difference = G[A][B] - G[A][C] + G[B][C]
                    difference_values.add(difference)

    return difference_values

def experiment_all_triangle_equality_graphs_exist():
    n = 4
    expected = set(multigraphs_satisfying_triangle_inequality(n,2))
    actual = set()
    for l in tqdm(range(11)):
        for w in permutations(n, l):
            actual.add(inversion_multigraph(w))
    return actual, expected

# print(experiment_difference_values())

actual, expected = experiment_all_triangle_equality_graphs_exist()
print(expected - actual)
print(len(expected))
print(len(actual))