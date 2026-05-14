# -*- coding: utf-8 -*-
r"""
最佳 PMU 分配 —— 布尔约束化简 + ILP 精确求解（Option B：超节点折叠）
====================================================================

数学模型（与 算法方案.md §三-B 严格一致）：

  原图 G = (V, E)，|V| = n = 37，邻接矩阵含自环 A_ii = 1。
  Z      : ZI 节点集 (|Z| = 11)
  x ∈ {0,1}^n : PMU 决策；x_i = 1 表示 i 装 PMU
  N[i]  : G 中 i 的闭邻域

  ── 拓扑折叠为超图 G' = (V', E') ──
    对每个 ZI 连通分量 C_p ⊆ Z 引入超节点 v_p：
      V'   = (V ∖ Z) ∪ {v_1, ..., v_P}
      原边 (u,w) 同属某 C_p → 删除
      原边 (u,w), u ∈ C_p, w ∉ Z → 替换为 (v_p, w)
      其余原边保留
      每个 v ∈ V' 含自环
    Z' = {v_1, ..., v_P}，每个 v_p 是单 ZI 节点 (k=1)。

  ── 提升 (lifting) ──
    x'_w   = x_w                  (w ∈ V ∖ Z)
    x'_{v_p} = ⋁_{z ∈ C_p} x_z   (超节点装 PMU ⇔ 内部某 z 装 PMU)

  ── G' 中可观函数 ──
    f'(v) = ⋁_{u ∈ N'[v]} x'_u
    回映到原变量：
      L(i) = (N[i] ∖ Z) ∪ ⋃_{p: N[i] ∩ C_p ≠ ∅} C_p     (i ∈ V ∖ Z)
      f'(i)   = ⋁_{l ∈ L(i)} x_l
      f'(v_p) = ⋁_{l ∈ S_{C_p}} x_l,  S_{C_p} = C_p ∪ E_{C_p}

  ── G' 上的单 ZI 松弛 (k=1)，对每个 v_p ──
    删除 f'(v_p) = 1
    对每个 j ∈ E_{C_p}：
      f'(j) ∨ ⋀_{u ∈ N'[v_p] ∖ {j}} f'(u) = 1
    (因 |N'[v_p] ∖ {j}| = m_p - k_p，仅一种 T 选择)

  目标：min Σ_i x_i  s.t. 上述约束。

布尔化简 & ILP 线性化：与原方案相同
  - SOP 上幂等 + 吸收律 → 最小 SOP
  - 多变量积项引入 0-1 辅助 y, y ≤ x_j；OR 主约束 Σ ≥ 1
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
# 4. 构造每个非 ZI 节点的最终约束 SOP（Option B：超节点折叠）
# ----------------------------------------------------------------------
def build_constraints(n, N, Z, edges):
    """
    Option B 约束生成（数学定义见模块顶部 docstring）。

    对每个 i ∈ V ∖ Z：
        初始 f'(i) = ⋁_{l ∈ L(i)} x_l                                   …(1)
        若存在 p 使 i ∈ E_{C_p}，对每个这样的 p 计算
            rule_p = f'(v_p) ∧ ⋀_{j ∈ E_{C_p} ∖ {i}} f'(j)              …(2)
        最终约束 = (1) OR_p (2)                                          …(3)

    ZI 节点 z ∈ Z 不再有独立约束（被折叠进 v_p）。

    返回:
        constraints : dict[int -> SOP]   仅 i ∈ V ∖ Z
        supers      : list[(C, S)]       兼容旧接口，喂给 verify
    """
    components = zi_components(Z, edges)           # list[frozenset]
    P_count = len(components)

    # comp_of[z] = 索引 p（z ∈ C_p）；其余为 None
    comp_of = {i: None for i in range(1, n + 1)}
    for p, C in enumerate(components):
        for z in C:
            comp_of[z] = p

    # E_{C_p} = N(C_p) ∖ C_p ⊆ V ∖ Z ； S_{C_p} = C_p ∪ E_{C_p}
    E_C_list = []
    S_C_list = []
    for C in components:
        ext = set()
        for z in C:
            ext |= N[z]                            # N[z] 含 z 自身
        ext -= C
        assert ext, f"ZI 连通分量 {sorted(C)} 没有外部邻居，模型不可解。"
        E_C_list.append(ext)
        S_C_list.append(set(C) | ext)

    # L(i) for i ∈ V ∖ Z
    #   L(i) = (N[i] ∖ Z) ∪ ⋃_{p: N[i] ∩ C_p ≠ ∅} C_p
    L = {}
    for i in range(1, n + 1):
        if i in Z:
            continue
        Li = set()
        for j in N[i]:                             # N[i] 含 i 自身
            if j in Z:
                Li |= components[comp_of[j]]       # ZI 邻居 → 整个分量
            else:
                Li.add(j)
        L[i] = Li
    # 健壮性：每个非 ZI 节点至少能"看见"自己
    for i in L:
        assert i in L[i], f"L({i}) 不含自身，闭邻域构造异常"

    # 初始 f'(i) SOP（仅 i ∈ V ∖ Z），每个 clause 为单变量
    constraints = {i: [frozenset([l]) for l in L[i]] for i in L}

    # 单 ZI 松弛：对每个 v_p ∈ Z'，对每个 i ∈ E_{C_p} 累加 rule_p
    for p in range(P_count):
        S = S_C_list[p]                            # f'(v_p) = ⋁_{l∈S} x_l
        E = E_C_list[p]
        f_vp_sop  = [frozenset([l]) for l in S]
        f_ext_sop = {j: [frozenset([l]) for l in L[j]] for j in E}

        for i in E:
            # rule_p^(i) = f'(v_p) ∧ ⋀_{j ∈ E ∖ {i}} f'(j)
            and_factors = [f_vp_sop] + [f_ext_sop[j] for j in E if j != i]
            rule_sop = sop_and_many(and_factors)
            # 若 i 同属多个 E_{C_p}，规则按 OR 累加；与初始 f'(i) 也 OR
            constraints[i] = sop_or(constraints[i], rule_sop)

    # 约束数与正确性自检
    assert len(constraints) == n - len(Z), \
        f"约束数 {len(constraints)} ≠ n-|Z| = {n - len(Z)}"
    for i, sop in constraints.items():
        assert sop, f"节点 {i} 的约束为常 FALSE，模型不可行。"

    supers = [(components[p], S_C_list[p]) for p in range(P_count)]
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
# 6. 解的验证：在超图 G' 上做不动点，再映回原图（Option B）
# ----------------------------------------------------------------------
def verify(x_val, N, supers):
    """
    Option B 验证。

    G' 节点 ID 约定：
        原非 ZI 节点 i ∈ V ∖ Z  → 整数 i
        超节点 v_p (p=0..P-1)   → 元组 ('super', p)

    步骤：
        1. 构造 G' 的 PMU 集 P'：
              x'_w     = x_w               (w ∈ V ∖ Z)
              x'_{v_p} = ⋁_{z ∈ C_p} x_z
              P' = {v ∈ V' : x'_v = 1}
        2. 初始可观: obs' = {v ∈ V' : N'[v] ∩ P' ≠ ∅}
        3. 不动点：对每个 v_p ∈ Z'
              若 |N'[v_p] ∩ obs'| ≥ m_p - k_p  ⇒  obs' ← obs' ∪ N'[v_p]
        4. 映回 V：
              i ∈ V ∖ Z ：i ∈ obs  ⇔  i ∈ obs'
              z ∈ C_p   ：z ∈ obs  ⇔  v_p ∈ obs'
    """
    n = len(x_val)
    placed = {i + 1 for i in range(n) if x_val[i] == 1}

    # 反推 Z / comp_of / E_C / S_C / components
    components = [set(C) for C, _ in supers]
    Z = set().union(*components) if components else set()
    comp_of = {i: None for i in range(1, n + 1)}
    for p, C in enumerate(components):
        for z in C:
            comp_of[z] = p
    E_C_list = [set(S) - C for C, S in zip(components, [S for _, S in supers])]
    S_C_list = [set(S) for _, S in supers]
    P_count = len(components)

    # 1) G' 中的 PMU 集 P'
    P_prime = set()
    for i in placed:
        if i in Z:
            P_prime.add(('super', comp_of[i]))     # x'_{v_p} = ⋁_{z∈C_p} x_z
        else:
            P_prime.add(i)

    # G' 闭邻域 N'[v]
    N_prime = {}
    for i in range(1, n + 1):
        if i in Z:
            continue
        nbr = set()
        for j in N[i]:                              # 含自环 j = i
            if j in Z:
                nbr.add(('super', comp_of[j]))
            else:
                nbr.add(j)
        N_prime[i] = nbr
        assert i in nbr, f"N'[{i}] 缺自身"
    for p in range(P_count):
        N_prime[('super', p)] = {('super', p)} | E_C_list[p]

    # 2) 初始 obs'
    obs_prime = {v for v, nb in N_prime.items() if nb & P_prime}

    # 3) 不动点：G' 上的单 ZI 规则
    while True:
        changed = False
        for p in range(P_count):
            v_p = ('super', p)
            threshold = len(S_C_list[p]) - len(components[p])   # m_p - k_p
            if len(N_prime[v_p] & obs_prime) >= threshold:
                new = N_prime[v_p] - obs_prime
                if new:
                    obs_prime |= new
                    changed = True
        if not changed:
            break

    # 4) 映回 V
    obs = set()
    for i in range(1, n + 1):
        if i in Z:
            if ('super', comp_of[i]) in obs_prime:
                obs.add(i)
        else:
            if i in obs_prime:
                obs.add(i)

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
    import os
    _OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "result.txt")
    with open(_OUT, "w", encoding="utf-8") as f:
        with contextlib.redirect_stdout(f):
            main()
