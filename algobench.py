"""AlgoBench core: algorithms + benchmarking harness (time, node counts, solution quality, empirical exponent)."""
import math, random, sys, time
from functools import lru_cache
import numpy as np
import pandas as pd

sys.setrecursionlimit(10000)

# ---- Sorting: O(n^2) vs O(n log n) -------------------------------------------
def bubble(a):
    a = a[:]; n = len(a)
    for i in range(n):
        for j in range(n - i - 1):
            if a[j] > a[j + 1]: a[j], a[j + 1] = a[j + 1], a[j]
    return a

def merge_sort(a):
    if len(a) < 2: return a
    m = len(a) // 2; l, r = merge_sort(a[:m]), merge_sort(a[m:]); o = []; i = j = 0
    while i < len(l) and j < len(r):
        if l[i] <= r[j]: o.append(l[i]); i += 1
        else: o.append(r[j]); j += 1
    return o + l[i:] + r[j:]

def quick(a):  # randomized pivot: expected O(n log n)
    if len(a) < 2: return a
    p = random.choice(a)
    return quick([x for x in a if x < p]) + [x for x in a if x == p] + quick([x for x in a if x > p])

# ---- Dynamic programming: Fibonacci ------------------------------------------
def fib_naive(n): return n if n < 2 else fib_naive(n - 1) + fib_naive(n - 2)   # O(2^n)

def fib_memo(n):                                                                # O(n) top-down
    @lru_cache(None)
    def f(k): return k if k < 2 else f(k - 1) + f(k - 2)
    return f(n)

def fib_tab(n):                                                                 # O(n) bottom-up, O(1) space
    a, b = 0, 1
    for _ in range(n): a, b = b, a + b
    return a

# ---- 0/1 Knapsack: brute force vs DP -----------------------------------------
def knap_brute(w, v, C, i=0):                                                   # O(2^n)
    if i == len(w): return 0
    best = knap_brute(w, v, C, i + 1)
    if w[i] <= C: best = max(best, v[i] + knap_brute(w, v, C - w[i], i + 1))
    return best

def knap_dp(w, v, C):                                                           # O(nC)
    dp = [0] * (C + 1)
    for wi, vi in zip(w, v):
        for c in range(C, wi - 1, -1): dp[c] = max(dp[c], dp[c - wi] + vi)
    return dp[C]

# ---- Graphs: BFS vs Dijkstra vs Bellman-Ford -----------------------------------
from collections import deque
import heapq
def rand_graph(V, E):
    g = [[] for _ in range(V)]
    for i in range(1, V): g[random.randrange(i)].append((i, random.randint(1, 20)))
    for _ in range(E - V + 1): g[random.randrange(V)].append((random.randrange(V), random.randint(1, 20)))
    return g
def bfs(g, s=0):                                   # O(V+E)
    d = [-1] * len(g); d[s] = 0; q = deque([s])
    while q:
        u = q.popleft()
        for v, _ in g[u]:
            if d[v] < 0: d[v] = d[u] + 1; q.append(v)
    return d
def dijkstra_heap(g, s=0):                         # O((V+E) log V), greedy
    d = [float("inf")] * len(g); d[s] = 0; h = [(0, s)]
    while h:
        du, u = heapq.heappop(h)
        if du > d[u]: continue
        for v, w in g[u]:
            if du + w < d[v]: d[v] = du + w; heapq.heappush(h, (d[v], v))
    return d
def bellman_ford(g, s=0):                          # O(VE), DP over path length
    d = [float("inf")] * len(g); d[s] = 0
    for _ in range(len(g)):
        ch = False
        for u in range(len(g)):
            for v, w in g[u]:
                if d[u] + w < d[v]: d[v] = d[u] + w; ch = True
        if not ch: break
    return d

# ---- Backtracking: N-Queens search-space reduction -----------------------------
def queens_nodes(n, prune):
    cnt = 0
    def ok(cols, c): return all(c != x and abs(c - x) != len(cols) - i for i, x in enumerate(cols))
    def rec(cols):
        nonlocal cnt; cnt += 1
        if len(cols) == n: return
        for c in range(n):
            if not prune or ok(cols, c): rec(cols + [c])
    rec([]); return cnt

# ---- Greedy knapsack + approximation guarantee ---------------------------------
def knap_greedy(w, v, C):
    t = 0
    for wi, vi in sorted(zip(w, v), key=lambda p: -p[1] / p[0]):
        if wi <= C: C -= wi; t += vi
    return t
def knap_greedy2(w, v, C):                         # greedy vs best single item: 1/2-approximation
    return max(knap_greedy(w, v, C), max([vi for wi, vi in zip(w, v) if wi <= C], default=0))

# ---- Memory profiling ----------------------------------------------------------
import tracemalloc
def peak_kb(f, *a):
    tracemalloc.start(); f(*a); _, pk = tracemalloc.get_traced_memory(); tracemalloc.stop(); return pk / 1024

# ---- Branch & bound (0/1 knapsack, fractional upper bound) ---------------------
def knap_nodes(w, v, C):
    c = 0
    def r(i, cap):
        nonlocal c; c += 1
        if i == len(w): return
        r(i + 1, cap)
        if w[i] <= cap: r(i + 1, cap - w[i])
    r(0, C); return c
def knap_bb(w, v, C):
    it = sorted(zip(w, v), key=lambda p: -p[1] / p[0]); n = len(it); best = 0; nodes = 0
    def bound(i, cap, val):
        for wi, vi in it[i:]:
            if wi <= cap: cap -= wi; val += vi
            else: return val + vi * cap / wi
        return val
    def rec(i, cap, val):
        nonlocal best, nodes; nodes += 1; best = max(best, val)
        if i == n or bound(i, cap, val) <= best: return
        if it[i][0] <= cap: rec(i + 1, cap - it[i][0], val + it[i][1])
        rec(i + 1, cap, val)
    rec(0, C, 0); return best, nodes

# ---- Max-flow: Edmonds-Karp (BFS) vs Ford-Fulkerson (DFS) -----------------------
def flow_graph(V, E):
    c = [{} for _ in range(V)]
    for _ in range(E):
        a, b = random.randrange(V - 1), random.randrange(1, V)
        if a != b: c[a][b] = c[a].get(b, 0) + random.randint(1, 20); c[b].setdefault(a, 0)
    return c
def _maxflow(c, bfs_mode):
    c = [dict(d) for d in c]; t = len(c) - 1; flow = 0
    while True:
        par = {0: None}; q = deque([0])
        while q and t not in par:
            u = q.popleft() if bfs_mode else q.pop()
            for v, cap in c[u].items():
                if cap > 0 and v not in par: par[v] = u; q.append(v)
        if t not in par: return flow
        b = float("inf"); v = t
        while par[v] is not None: b = min(b, c[par[v]][v]); v = par[v]
        v = t
        while par[v] is not None:
            u = par[v]; c[u][v] -= b; c[v][u] = c[v].get(u, 0) + b; v = u
        flow += b
def edmonds_karp(c): return _maxflow(c, True)
def ford_fulkerson_dfs(c): return _maxflow(c, False)

# ---- Approximation: vertex cover -----------------------------------------------
def vc_graph(n, p=0.3): return [(i, j) for i in range(n) for j in range(i + 1, n) if random.random() < p]
def vc_opt(n, E):
    m = [(1 << a) | (1 << b) for a, b in E]
    return min(bin(s).count("1") for s in range(1 << n) if all(s & e for e in m))
def vc_matching(n, E):
    u = set(); k = 0
    for a, b in E:
        if a not in u and b not in u: u |= {a, b}; k += 2
    return k
def vc_greedy(n, E):
    E = list(E); k = 0
    while E:
        d = [0] * n
        for a, b in E: d[a] += 1; d[b] += 1
        v = d.index(max(d)); E = [e for e in E if v not in e]; k += 1
    return k

# ---- Amortized analysis: dynamic array growth policies -------------------------
def amortized_cost(n, grow):
    cap = 1; size = 0; copies = 0
    for _ in range(n):
        if size == cap: copies += size; cap = grow(cap)
        size += 1
    return 1 + copies / n

# ---- Minimum spanning tree: Kruskal (union-find) vs Prim (heap) -----------------
def mst_graph(V, E):
    ed = [(random.randint(1, 100), i, random.randrange(i)) for i in range(1, V)]
    ed += [(random.randint(1, 100), random.randrange(V), random.randrange(V)) for _ in range(E - V + 1)]
    return V, [e for e in ed if e[1] != e[2]]
def kruskal(g):                                    # O(E log E)
    V, ed = g; p = list(range(V))
    def f(x):
        while p[x] != x: p[x] = p[p[x]]; x = p[x]
        return x
    t = 0
    for w, a, b in sorted(ed):
        ra, rb = f(a), f(b)
        if ra != rb: p[ra] = rb; t += w
    return t
def prim(g):                                       # O(E log V)
    V, ed = g; adj = [[] for _ in range(V)]
    for w, a, b in ed: adj[a].append((w, b)); adj[b].append((w, a))
    seen = [False] * V; h = [(0, 0)]; t = 0
    while h:
        w, u = heapq.heappop(h)
        if seen[u]: continue
        seen[u] = True; t += w
        for e in adj[u]:
            if not seen[e[1]]: heapq.heappush(h, e)
    return t

# ---- TSP: exhaustive search vs branch & bound (nearest-neighbour upper bound) ---
def tsp_instance(n):
    pts = [(random.random(), random.random()) for _ in range(n)]
    return [[math.dist(a, b) for b in pts] for a in pts]
def tsp_search(d, prune):
    n = len(d); best = [float("inf")]; nodes = [0]
    if prune:
        cur = 0; seen = {0}; L = 0
        while len(seen) < n:
            nx = min((j for j in range(n) if j not in seen), key=lambda j: d[cur][j]); L += d[cur][nx]; seen.add(nx); cur = nx
        best[0] = L + d[cur][0]
    def rec(c, seen, L):
        nodes[0] += 1
        if prune and L >= best[0]: return
        if len(seen) == n: best[0] = min(best[0], L + d[c][0]); return
        for j in range(n):
            if j not in seen: rec(j, seen | {j}, L + d[c][j])
    rec(0, {0}, 0.0); return best[0], nodes[0]

# ---- LCS: naive recursion vs DP table --------------------------------------------
def lcs_calls(a, b):
    c = 0
    def f(i, j):
        nonlocal c; c += 1
        if i == len(a) or j == len(b): return 0
        return 1 + f(i + 1, j + 1) if a[i] == b[j] else max(f(i + 1, j), f(i, j + 1))
    f(0, 0); return c
def lcs_len(a, b):
    T = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            T[i][j] = T[i-1][j-1] + 1 if a[i-1] == b[j-1] else max(T[i-1][j], T[i][j-1])
    return T[-1][-1]

# ---- Harness ------------------------------------------------------------------
def timeit(f, *a, reps=3):
    best = float("inf")
    for _ in range(reps):
        t = time.perf_counter(); f(*a); best = min(best, time.perf_counter() - t)
    return best * 1000  # ms

SUITES = ["Sorting", "Fibonacci (DP)", "Knapsack (DP vs brute force)",
          "Graphs: BFS vs Dijkstra vs Bellman-Ford", "Backtracking: N-Queens pruning", "Greedy vs optimal (quality)", "Memory: sorting peak KB",
          "Branch & bound: knapsack nodes", "Network flow: Edmonds-Karp vs DFS",
          "Approximation: vertex cover ratio", "Amortized: dynamic array cost",
          "Minimum spanning tree: Kruskal vs Prim", "TSP: branch & bound nodes", "LCS: recursion vs DP table"]
UNITS = {SUITES[0]: "milliseconds", SUITES[1]: "milliseconds", SUITES[2]: "milliseconds",
         SUITES[3]: "milliseconds", SUITES[4]: "nodes explored", SUITES[5]: "% of optimal value",
         SUITES[6]: "peak KB", SUITES[7]: "nodes explored", SUITES[8]: "milliseconds",
         SUITES[9]: "approx ratio (size / OPT)", SUITES[10]: "amortized cost per append",
         SUITES[11]: "milliseconds", SUITES[12]: "nodes explored", SUITES[13]: "subproblem evaluations"}

def suite(name):
    rows = []
    if name == "Sorting":
        for n in [250, 500, 1000, 2000, 4000]:
            a = [random.randint(0, 10**6) for _ in range(n)]
            rows += [dict(n=n, algo=f.__name__, value=timeit(f, a)) for f in (bubble, merge_sort, quick)]
    elif name.startswith("Fib"):
        for n in [10, 15, 20, 25, 30]:
            rows += [dict(n=n, algo=f.__name__, value=timeit(f, n)) for f in (fib_naive, fib_memo, fib_tab)]
    elif name.startswith("Knap"):
        for n in [8, 10, 12, 14, 16, 18]:
            w = [random.randint(5, 30) for _ in range(n)]; v = [random.randint(10, 100) for _ in range(n)]
            rows += [dict(n=n, algo="knap_brute", value=timeit(knap_brute, w, v, 100)),
                     dict(n=n, algo="knap_dp", value=timeit(knap_dp, w, v, 100))]
    elif name.startswith("Graphs"):
        for V in [200, 400, 800, 1600, 3200]:
            g = rand_graph(V, 4 * V)
            rows += [dict(n=V, algo=f.__name__, value=timeit(f, g)) for f in (bfs, dijkstra_heap, bellman_ford)]
    elif name.startswith("Back"):
        for n in [4, 5, 6, 7]:
            rows += [dict(n=n, algo="backtracking", value=queens_nodes(n, True)),
                     dict(n=n, algo="no_pruning", value=queens_nodes(n, False))]
    elif name.startswith("Greedy"):
        for n in [10, 20, 30, 40, 50]:
            r = {"greedy": 0, "greedy_plus_best_item": 0}
            for _ in range(30):
                w = [random.randint(5, 30) for _ in range(n)]; v = [random.randint(10, 100) for _ in range(n)]
                C = sum(w) // 3; o = knap_dp(w, v, C)
                r["greedy"] += 100 * knap_greedy(w, v, C) / o; r["greedy_plus_best_item"] += 100 * knap_greedy2(w, v, C) / o
            rows += [dict(n=n, algo=k, value=x / 30) for k, x in r.items()]
    elif name.startswith("Memory"):
        for n in [500, 1000, 2000, 4000, 8000]:
            a = [random.randint(0, 10**6) for _ in range(n)]
            rows += [dict(n=n, algo=f.__name__, value=peak_kb(f, a)) for f in (bubble, merge_sort, quick)]
    elif name.startswith("Branch"):
        for n in [10, 12, 14, 16, 18]:
            w = [random.randint(5, 30) for _ in range(n)]; v = [random.randint(10, 100) for _ in range(n)]; C = sum(w) // 3
            best, nodes = knap_bb(w, v, C); assert best == knap_dp(w, v, C)
            rows += [dict(n=n, algo="brute_force", value=knap_nodes(w, v, C)), dict(n=n, algo="branch_and_bound", value=nodes)]
    elif name.startswith("Network"):
        for V in [50, 100, 200, 400, 800]:
            c = flow_graph(V, 5 * V); assert edmonds_karp(c) == ford_fulkerson_dfs(c)
            rows += [dict(n=V, algo=f.__name__, value=timeit(f, c, reps=2)) for f in (edmonds_karp, ford_fulkerson_dfs)]
    elif name.startswith("Approx"):
        for n in [6, 8, 10, 12, 14]:
            r = {"matching_2approx": 0, "greedy_max_degree": 0}; k = 0
            for _ in range(15):
                E = vc_graph(n)
                if not E: continue
                o = vc_opt(n, E); k += 1
                r["matching_2approx"] += vc_matching(n, E) / o; r["greedy_max_degree"] += vc_greedy(n, E) / o
            rows += [dict(n=n, algo=a, value=x / k) for a, x in r.items()]
    elif name.startswith("Amortized"):
        for n in [100, 1000, 10000, 100000]:
            for nm, g in [("doubling", lambda c: 2 * c), ("grow_1.5x", lambda c: max(c + 1, int(c * 1.5))), ("grow_plus_1", lambda c: c + 1)]:
                rows.append(dict(n=n, algo=nm, value=amortized_cost(n, g)))
    elif name.startswith("Minimum"):
        for V in [200, 400, 800, 1600, 3200]:
            g = mst_graph(V, 4 * V); assert kruskal(g) == prim(g)
            rows += [dict(n=V, algo=f.__name__, value=timeit(f, g)) for f in (kruskal, prim)]
    elif name.startswith("TSP"):
        for n in [5, 6, 7, 8, 9]:
            d = tsp_instance(n); b0, n0 = tsp_search(d, False); b1, n1 = tsp_search(d, True); assert abs(b0 - b1) < 1e-9
            rows += [dict(n=n, algo="brute_force", value=n0), dict(n=n, algo="branch_and_bound", value=n1)]
    else:  # LCS
        for n in [4, 6, 8, 10, 12]:
            a = "".join(random.choice("ACGT") for _ in range(n)); b = "".join(random.choice("ACGT") for _ in range(n))
            assert lcs_len(a, b) <= n
            rows += [dict(n=n, algo="naive_recursion", value=lcs_calls(a, b)), dict(n=n, algo="dp_table", value=(n + 1) ** 2)]
    return pd.DataFrame(rows)

def exponents(df):
    """Empirical growth exponent k from a log-log fit t ~ n^k (meaningful for polynomial algorithms)."""
    out = {}
    for algo, g in df.groupby("algo"):
        g = g[g.value > 0]
        out[algo] = round(float(np.polyfit(np.log(g.n), np.log(g.value), 1)[0]), 2) if len(g) > 2 else None
    return pd.Series(out, name="empirical exponent k")

# ---- Theory vs empirical: least-squares fit y = c*f(n), reported as R^2 --------
MODELS = {"bubble": lambda n: n**2, "merge_sort": lambda n: n*np.log2(n), "quick": lambda n: n*np.log2(n),
          "fib_naive": lambda n: 1.618**n, "fib_memo": lambda n: n, "fib_tab": lambda n: n,
          "knap_brute": lambda n: 2.0**n, "knap_dp": lambda n: n,
          "bfs": lambda n: n, "kruskal": lambda n: n*np.log2(n), "prim": lambda n: n*np.log2(n), "dijkstra_heap": lambda n: n*np.log2(n), "bellman_ford": lambda n: n**2}

def theory_fit(df):
    out = {}
    for a, g in df.groupby("algo"):
        if a not in MODELS: continue
        n = g.n.values.astype(float); y = g.value.values; f = MODELS[a](n); c = (y*f).sum() / (f*f).sum()
        out[a] = round(1 - ((y - c*f)**2).sum() / ((y - y.mean())**2).sum(), 4)
    return pd.Series(out, name="R2 vs theoretical model")
