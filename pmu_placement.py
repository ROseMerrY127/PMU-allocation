# -*- coding: utf-8 -*-
r"""
最佳 PMU 分配 —— 布尔约束化简 + ILP 精确求解
==============================================

变量映射（与 算法方案.md / 最佳PMU分配.md 一致）：
    n            : 节点总数 (= 37)
    A_ij         : 邻接矩阵元素；含自环 A_ii = 1
    x_i in {0,1} : 决策变量；x_i = 1 表示 i 节点装 PMU
    N[i]         : 节点 i 的闭邻域 = {j : A_ij = 1}
    Z            : 特殊（零注入）节点集
    f_i = OR_{j in N[i]} x_j    : 节点 i 的可观布尔函数
    S_C = C ∪ U_{z in C} N(z)   : ZI 群 C 的超群
    k = |C|, m = |S_C|          : ZI 群规模与超群规模

数学模型：
    min  Sum_i x_i
    s.t. f_i = 1 for all i      (经 ZI 松弛后修改约束)

ZI 松弛规则（合并 ZI 群策略）：
    对每个 ZI 群 C：
      - 删除 f_z = 1, for all z in C
      - 对每个 i in S_C \ C：
            f_i  OR  OR_{T subset S_C\{i}, |T|=m-k}  AND_{j in T} f_j  =  1

布尔化简：
    单调布尔函数的最小 SOP = 极小蕴含项集合，仅需 [幂等 + 吸收] 即可。

ILP 线性化：
    对化简后约束 OR_k AND_{j in P_ik} x_j >= 1
      - 单变量子句 {j}: 直接保留 x_j
      - 多变量子句 P : 引入 0-1 辅助 y, 加 y <= x_j (for all j in P)
      - 主约束       : Sum_k term_k >= 1
"""
import sys, io
# 强制 stdout 用 UTF-8，避免 Windows GBK 控制台乱码
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from itertools import combinations
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import lil_matrix


# ----------------------------------------------------------------------
# 0. 输入：边集合（来自 最佳PMU分配.md 末尾）与特殊节点
# ----------------------------------------------------------------------
EDGES = [
    (1, 4), (2, 6), (3, 8), (4, 9), (5, 6), (6, 12), (7, 8),
    (8, 9), (9, 10), (9, 18), (10, 11), (11, 13), (11, 12), (12, 14),
    (13, 19), (14, 20), (15, 16), (16, 17), (16, 21), (17, 18), (18, 22),
    (22, 25), (23, 24), (24, 25), (24, 27), (25, 26), (25, 28), (27, 31),
    (29, 30), (30, 31), (30, 32), (31, 33), (33, 34), (34, 35), (35, 36), (35, 37),
]
ZI_NODES = {6, 8, 9, 11, 14, 18, 24, 25, 28, 30, 35}


# ----------------------------------------------------------------------
# 1. 拓扑：构造闭邻域 N[i]（A_ii = 1）
# ----------------------------------------------------------------------
def build_neighbors(edges):
    nodes = set()
    for u, v in edges:
        nodes.add(u); nodes.add(v)
    n = max(nodes)
    missing = set(range(1, n + 1)) - nodes
    assert not missing, f"节点编号不连续，缺失: {sorted(missing)}"
    N = {i: {i} for i in range(1, n + 1)}      # A_ii = 1（自环）
    for u, v in edges:
        N[u].add(v); N[v].add(u)
    return n, N


# ----------------------------------------------------------------------
# 2. 识别 ZI 连通分量（在原图诱导子图 G[Z] 上做并查集）
# ----------------------------------------------------------------------
def zi_components(Z, edges):
    parent = {z: z for z in Z}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    for u, v in edges:
        if u in Z and v in Z:
            union(u, v)
    comps = {}
    for z in Z:
        comps.setdefault(find(z), set()).add(z)
    return [frozenset(c) for c in comps.values()]


# ----------------------------------------------------------------------
# 3. SOP（Sum-Of-Products）的布尔运算
#    每个 SOP 表示为 list[frozenset[int]]，含义：⋁ clause(⋀ x_j)
#    空列表 = 常 FALSE；[frozenset()] = 常 TRUE
# ----------------------------------------------------------------------
def absorb(clauses):
    """
    应用幂等 + 吸收律到极小化形式（对单调布尔函数即最小 SOP）：
        a + a    -> a   （幂等：通过去重实现）
        a + ab   -> a   （吸收：丢弃任何被真子集吸收的 clause）
    """
    uniq = list({frozenset(c) for c in clauses})  # 去重 & 转 frozenset
    uniq.sort(key=len)
    keep = []
    for c in uniq:
        if not any(r < c for r in keep):          # 真子集已被保留 → 被吸收
            keep.append(c)
    return keep


def sop_or(A, B):
    return absorb(A + B)


def sop_and(A, B):
    """SOP × SOP → SOP，clause 之间两两并集（积分配）。"""
    if not A or not B:
        return []                                  # FALSE 吸收一切
    return absorb([a | b for a in A for b in B])


def sop_and_many(sops):
    if not sops:
        return [frozenset()]                       # 空 AND = TRUE
    res = sops[0]
    for s in sops[1:]:
        res = sop_and(res, s)
        if not res:
            return []
    return res


# ----------------------------------------------------------------------
# 4. 构造每个节点的最终约束 SOP
# ----------------------------------------------------------------------
def build_constraints(n, N, Z, edges):
    """
    返回:
        constraints : dict[int -> SOP]   每个保留节点的最终 SOP 约束（=1）
        supers      : list[(C, S_C)]     ZI 超群（用于解的事后验证）
    """
    # 初始 f_i^0 = ⋁_{j ∈ N[i]} x_j   （单元素 clause 的 SOP）
    f0 = {i: [frozenset([j]) for j in N[i]] for i in range(1, n + 1)}

    # ZI 连通分量 → 超群
    components = zi_components(Z, edges)
    supers = []
    for C in components:
        S = set(C)
        for z in C:
            S |= N[z]
        supers.append((C, S))

    # 每个节点的当前约束：先复制 f0
    constraints = {i: list(f0[i]) for i in range(1, n + 1)}

    # 步骤 (i)：删除每个 ZI 节点 z 的等式 f_z = 1
    for C, S in supers:
        for z in C:
            constraints.pop(z, None)

    # 步骤 (ii)：对每个 i ∈ S_C \ C，OR 上规则项
    #   ⋁_{T ⊆ S_C\{i}, |T| = m-k} ⋀_{j∈T} f_j^0
    for C, S in supers:
        k = len(C)
        m = len(S)
        T_size = m - k
        S_sorted = sorted(S)
        for i in S:
            if i in C:
                continue                          # ZI 节点已删
            others = [j for j in S_sorted if j != i]
            rule_sop = []                         # 起始 = FALSE
            for T in combinations(others, T_size):
                T_sop = sop_and_many([f0[j] for j in T])
                rule_sop = sop_or(rule_sop, T_sop)
            # 多个超群规则按 OR 累加（节点同时属于多个超群时）
            constraints[i] = sop_or(constraints[i], rule_sop)

    return constraints, supers


# ----------------------------------------------------------------------
# 5. 把 SOP 约束线性化为 ILP 并求解
# ----------------------------------------------------------------------
def solve_ilp(n, constraints):
    """
    决策变量布局:
      索引 0..n-1   : x_1..x_n     (Binary)
      索引 n..n+K-1 : y_1..y_K     (Binary, 多变量积项的辅助变量)
    """
    var_count = n
    or_terms_per_constraint = []   # 每条约束的 (变量索引列表)
    aux_pairs = []                 # (y_idx, x_idx)：表示 y_idx ≤ x_idx

    for i, sop in constraints.items():
        if not sop:
            raise RuntimeError(f"节点 {i} 的约束化为常 FALSE，模型不可行。")
        terms = []
        for clause in sop:
            if len(clause) == 1:
                (j,) = tuple(clause)
                terms.append(j - 1)               # 复用 x_j
            else:
                y_idx = var_count
                var_count += 1
                for j in clause:
                    aux_pairs.append((y_idx, j - 1))
                terms.append(y_idx)
        or_terms_per_constraint.append(terms)

    nvars = var_count
    # 目标 c：仅 x_i 计入
    c = np.zeros(nvars)
    c[:n] = 1.0

    # 线性约束矩阵
    n_or = len(or_terms_per_constraint)
    n_aux = len(aux_pairs)
    A = lil_matrix((n_or + n_aux, nvars))
    lb = np.empty(n_or + n_aux)
    ub = np.empty(n_or + n_aux)

    # OR 行：Σ terms ≥ 1
    for r, terms in enumerate(or_terms_per_constraint):
        for t in terms:
            A[r, t] = A[r, t] + 1.0
        lb[r] = 1.0
        ub[r] = np.inf

    # 辅助行：y - x_j ≤ 0
    for ridx, (y_idx, x_idx) in enumerate(aux_pairs):
        r = n_or + ridx
        A[r, y_idx] = 1.0
        A[r, x_idx] = -1.0
        lb[r] = -np.inf
        ub[r] = 0.0

    constraints_obj = LinearConstraint(A.tocsr(), lb, ub)
    integrality = np.ones(nvars)                  # 全二进制
    bounds = Bounds(np.zeros(nvars), np.ones(nvars))

    res = milp(c, constraints=constraints_obj,
               integrality=integrality, bounds=bounds)
    if not res.success:
        raise RuntimeError(f"MILP 求解失败：{res.message}")

    x_val = (res.x[:n] > 0.5).astype(int)
    return int(round(res.fun)), x_val, n_or, n_aux


# ----------------------------------------------------------------------
# 6. 解的验证：模拟 ZI 推断到不动点，检查全节点可观
# ----------------------------------------------------------------------
def verify(x_val, N, supers):
    """
    输入 x_val (0/1 长度 n)，沿 ZI 规则迭代到不动点，检查每个节点是否可观。
    """
    n = len(x_val)
    placed = {i + 1 for i in range(n) if x_val[i] == 1}

    # 直接可观：i 的闭邻域内有 PMU
    obs = {i for i in range(1, n + 1) if any(j in placed for j in N[i])}

    # ZI 不动点传播
    while True:
        changed = False
        for C, S in supers:
            k = len(C); m = len(S)
            obs_count = sum(1 for j in S if j in obs)
            if obs_count >= m - k:
                for j in S:
                    if j not in obs:
                        obs.add(j)
                        changed = True
        if not changed:
            break

    return obs == set(range(1, n + 1)), placed, obs


# ----------------------------------------------------------------------
# 7. 主程序
# ----------------------------------------------------------------------
def main():
    n, N = build_neighbors(EDGES)
    print(f"[INFO] 节点数 n = {n}；ZI 节点 = {sorted(ZI_NODES)}")

    components = zi_components(ZI_NODES, EDGES)
    print(f"[INFO] ZI 连通分量数 = {len(components)}：")
    for C in components:
        S = set(C)
        for z in C: S |= N[z]
        print(f"        C={sorted(C)}  k={len(C)}  S={sorted(S)}  m={len(S)}  m-k={len(S)-len(C)}")

    constraints, supers = build_constraints(n, N, ZI_NODES, EDGES)
    print(f"[INFO] 保留约束数 = {len(constraints)}（删去 {n - len(constraints)} 个 ZI 节点约束）")

    total_clauses = sum(len(sop) for sop in constraints.values())
    multi_clauses = sum(1 for sop in constraints.values() for cl in sop if len(cl) > 1)
    max_clause_len = max((len(cl) for sop in constraints.values() for cl in sop), default=0)
    print(f"[INFO] 化简后总 clause 数 = {total_clauses}，"
          f"多变量 clause 数 = {multi_clauses}，最大 clause 长度 = {max_clause_len}")

    obj, x_val, n_or, n_aux = solve_ilp(n, constraints)
    placed = sorted(i + 1 for i in range(n) if x_val[i] == 1)
    print(f"\n[ILP] OR 约束行 = {n_or}，辅助行 = {n_aux}")
    print(f"[ILP] 最优 PMU 数 = {obj}")
    print(f"[ILP] PMU 放置位置 = {placed}")

    ok, _, obs = verify(x_val, N, supers)
    print(f"\n[CHK] 可观验证：全 {n} 节点可观 = {ok}")
    if not ok:
        print(f"[CHK] 不可观节点 = {sorted(set(range(1, n+1)) - obs)}")


if __name__ == "__main__":
    main()
    # 同时把结果写到 result.txt 便于阅读（避免控制台编码问题）
    import contextlib
    with open("c:\\Users\\ASUS\\Desktop\\新建文件夹\\result.txt", "w", encoding="utf-8") as f:
        with contextlib.redirect_stdout(f):
            main()
