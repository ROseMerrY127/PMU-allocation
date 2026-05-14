# -*- coding: utf-8 -*-
r"""
鲁棒最优 PMU 分配 —— 单 PMU 失效下的 max-min 可观节点
======================================================

依据：鲁棒最优PMU分配方案.md（已用户二次确认）
ZI 处理：与 pmu_placement.py 同步采用 Option B（超节点折叠 + G' 单 ZI 规则）

数学符号（与 pmu_placement.py 严格一致）
    n              : 节点总数 (= 37)
    V = {1..n}     : 节点集
    N[i]           : 节点 i 的闭邻域（含自环）
    Z              : ZI 节点集
    C_p, E_{C_p}   : ZI 连通分量及其外邻；v_p 折叠超节点
    G' = (V', E')  : 折叠后的超图  (V' = (V∖Z) ∪ {v_p})
    x ∈ {0,1}^n    : 原 PMU 放置；x'_{v_p} = ⋁_{z∈C_p} x_z (提升)
    P(x)           : {i ∈ V : x_i = 1}
    Obs(P)         : 在 G' 中由闭邻域直接可观 + 单 ZI 规则
                     (|N'[v_p] ∩ obs'| ≥ m_p - k_p ⇒ N'[v_p] ⊆ obs')
                     不动点传播后，再映回 V 的可观节点集
                     (v_p ∈ obs' ⇒ C_p ⊆ Obs)
    k*             : 最少 PMU 数（Option B 下 ILP 求出）
    X*             : { x | Option-B 约束(x)=1, Σ x_i = k* }
    Surv(x, p)     : |Obs(P(x) ∖ {p})|
    R_min(x)       : min_{p ∈ P(x)} Surv(x, p)
    X**            : argmax_{x ∈ X*} R_min(x)

阶段
    I.  枚举 X*：在 ILP 中加 Σ x_i = k*，循环 + no-good cut
    II. 评分 Surv / R_min / R_avg
    III.筛选 X**（max R_min），全部输出

关键约束（线性化形式）
    Σ x_i        = k*                            （阶段 I 等式）
    Σ_{i ∈ P^(t)} x_i ≤ k* - 1                    （第 t 个解的 no-good cut）
"""
import sys, io, contextlib
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import lil_matrix, csr_matrix, vstack

from pmu_placement import (
    EDGES, ZI_NODES,
    build_neighbors, build_constraints, solve_ilp, verify,
)


# ----------------------------------------------------------------------
# 0. 参数
# ----------------------------------------------------------------------
MAX_ENUM = 500          # |X*| 截断阈值（议题 5）
import os as _os
OUTPUT_TXT = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                           "result_robust.txt")


# ----------------------------------------------------------------------
# 1. 把 SOP 约束编译成 ILP 基础矩阵（与 solve_ilp 等价但暴露中间结构）
# ----------------------------------------------------------------------
def build_base_ilp(n, constraints):
    """
    返回:
      base_A : csr_matrix, 形状 (n_or + n_aux, nvars)
      base_lb, base_ub : 行下/上界
      nvars  : 总变量数（含辅助 y）
      n_or   : OR 主约束行数
      n_aux  : 辅助行数 (y ≤ x_j)

    与 pmu_placement.solve_ilp 的建模规则严格一致：
      索引 0..n-1   : x_1..x_n
      索引 n..nvars-1 : 多变量积项辅助 y
    """
    var_count = n
    or_terms_per_constraint = []   # 每条约束: 该 OR 中各项的变量索引列表
    aux_pairs = []                 # (y_idx, x_idx)：y ≤ x_j

    for i, sop in constraints.items():
        if not sop:
            raise RuntimeError(f"节点 {i} 的约束化为常 FALSE，模型不可行。")
        terms = []
        for clause in sop:
            if len(clause) == 1:
                (j,) = tuple(clause)
                terms.append(j - 1)               # 单变量复用 x_j
            else:
                y_idx = var_count
                var_count += 1
                for j in clause:
                    aux_pairs.append((y_idx, j - 1))
                terms.append(y_idx)
        or_terms_per_constraint.append(terms)

    nvars = var_count
    n_or  = len(or_terms_per_constraint)
    n_aux = len(aux_pairs)

    A = lil_matrix((n_or + n_aux, nvars))
    lb = np.empty(n_or + n_aux)
    ub = np.empty(n_or + n_aux)

    # OR 主约束: Σ terms ≥ 1
    for r, terms in enumerate(or_terms_per_constraint):
        for t in terms:
            A[r, t] = A[r, t] + 1.0
        lb[r] = 1.0
        ub[r] = np.inf

    # 辅助约束: y - x_j ≤ 0
    for ridx, (y_idx, x_idx) in enumerate(aux_pairs):
        r = n_or + ridx
        A[r, y_idx] = 1.0
        A[r, x_idx] = -1.0
        lb[r] = -np.inf
        ub[r] = 0.0

    return A.tocsr(), lb, ub, nvars, n_or, n_aux


# ----------------------------------------------------------------------
# 2. 阶段 I：枚举 X*  (Σ x_i = k*  +  no-good cuts)
# ----------------------------------------------------------------------
def enumerate_optimal(n, constraints, k_star, max_enum=MAX_ENUM):
    r"""
    数学：循环求解
        find x ∈ {0,1}^nvars
        s.t.   ZI-SOP 约束(x) = 1
               Σ_{i=1..n} x_i = k_star
               Σ_{i ∈ P^(t)} x_i ≤ k_star - 1   (∀ 已得解 t)

    返回: (X_star, truncated)
      X_star    : list of np.ndarray (长度 n, dtype=int, ∈ {0,1})
      truncated : bool, 是否触及 max_enum
    """
    assert k_star >= 1
    base_A, base_lb, base_ub, nvars, _, _ = build_base_ilp(n, constraints)

    # 等式 Σ x_i = k_star （仅前 n 个变量参与）
    eq_row = lil_matrix((1, nvars))
    for j in range(n):
        eq_row[0, j] = 1.0
    eq_row = eq_row.tocsr()

    # 求解器公共参数
    c            = np.zeros(nvars)              # 可行性求解：目标 = 0
    integrality  = np.ones(nvars)               # 全二进制
    bounds       = Bounds(np.zeros(nvars), np.ones(nvars))

    X_star    = []
    cut_rows  = []                              # 每行: csr_matrix(1, nvars)
    cut_lb    = []
    cut_ub    = []
    truncated = False

    while True:
        if len(X_star) >= max_enum:
            truncated = True
            break

        # 拼装当前完整约束矩阵
        rows = [base_A, eq_row]
        lb_l = [base_lb, np.array([k_star])]
        ub_l = [base_ub, np.array([k_star])]
        if cut_rows:
            rows.append(vstack(cut_rows))
            lb_l.append(np.array(cut_lb))
            ub_l.append(np.array(cut_ub))

        full_A  = vstack(rows).tocsr()
        full_lb = np.concatenate(lb_l)
        full_ub = np.concatenate(ub_l)

        constr = LinearConstraint(full_A, full_lb, full_ub)
        res = milp(c, constraints=constr,
                   integrality=integrality, bounds=bounds)

        if not res.success:
            # 视作不可行 / 无更多解
            break

        x_val = (res.x[:n] > 0.5).astype(int)

        # 完整性断言：必须满足 |P| = k*
        assert int(x_val.sum()) == k_star, \
            f"枚举出的解 |P|={int(x_val.sum())} ≠ k*={k_star}"

        # 必须与已枚举解全不重复（cut 正确性自检）
        for prev in X_star:
            assert not np.array_equal(prev, x_val), "no-good cut 失效：解重复"

        X_star.append(x_val)

        # 进度日志：每枚举到 10 的整数倍个最优解，打印当前计数
        if len(X_star) % 10 == 0:
            print(f"[ENUM] 已找到 {len(X_star)} 个最优解 ...", flush=True)

        # 添加 no-good cut: Σ_{j ∈ P} x_j ≤ k_star - 1
        P_idx = [j for j in range(n) if x_val[j] == 1]
        new_cut = lil_matrix((1, nvars))
        for j in P_idx:
            new_cut[0, j] = 1.0
        cut_rows.append(new_cut.tocsr())
        cut_lb.append(-np.inf)
        cut_ub.append(float(k_star - 1))

    return X_star, truncated


# ----------------------------------------------------------------------
# 3. 阶段 II：对每个 x ∈ X*，评估单 PMU 失效的鲁棒性
# ----------------------------------------------------------------------
def evaluate_robustness(X_star, N, supers):
    r"""
    对每个 x：
        P = P(x)                        （PMU 节点编号，1-based 升序）
        Surv[t] = |Obs(P \ {P[t]})|     （t = 0..k*-1）
        R_min   = min Surv,  R_avg = mean Surv
        weakest = { P[t] : Surv[t] = R_min }
        weakest_unobs[p] = V \ Obs(P \ {p})

    返回: list[dict]，与 X_star 顺序一致。
    """
    n = len(next(iter(N.values())))  # 仅占位；下面用 max(N) 更稳
    n = max(N.keys())
    full_set = set(range(1, n + 1))

    results = []
    for x_val in X_star:
        assert len(x_val) == n
        P = sorted(j + 1 for j in range(n) if x_val[j] == 1)
        surv = []
        for p in P:
            x_fail = x_val.copy()
            x_fail[p - 1] = 0
            _, _, obs = verify(x_fail, N, supers)
            surv.append(len(obs))

        R_min = min(surv)
        R_avg = sum(surv) / len(surv)

        # 最脆弱 PMU 与其失效后的不可观节点集
        weakest = [P[t] for t in range(len(P)) if surv[t] == R_min]
        weakest_detail = []
        for p in weakest:
            x_fail = x_val.copy()
            x_fail[p - 1] = 0
            _, _, obs = verify(x_fail, N, supers)
            weakest_detail.append((p, sorted(full_set - obs)))

        results.append({
            "P":              P,
            "surv":           surv,
            "R_min":          R_min,
            "R_avg":          R_avg,
            "weakest":        weakest_detail,   # list[(p, [unobs nodes])]
        })

    return results


# ----------------------------------------------------------------------
# 4. 阶段 III：筛 X** = argmax R_min
# ----------------------------------------------------------------------
def select_robust_optimal(scored):
    R_best = max(s["R_min"] for s in scored)
    X_double = [s for s in scored if s["R_min"] == R_best]
    return R_best, X_double


# ----------------------------------------------------------------------
# 5. 主流程
# ----------------------------------------------------------------------
def main():
    # ---- 阶段 0：复用现行流程获得 k* 与所有结构 ----
    n, N = build_neighbors(EDGES)
    constraints, supers = build_constraints(n, N, ZI_NODES, EDGES)

    k_star, x0, _, _ = solve_ilp(n, constraints)
    print(f"[INFO] 节点数 n = {n}；k* = {k_star}")
    print(f"[INFO] 现行最优样例 PMU = {sorted(j+1 for j in range(n) if x0[j]==1)}")

    # ---- 阶段 I：枚举 X* ----
    print(f"\n[ENUM] 开始枚举所有 |P|=k* 的可行解 (MAX_ENUM={MAX_ENUM}) ...")
    X_star, truncated = enumerate_optimal(n, constraints, k_star, MAX_ENUM)
    print(f"[ENUM] |X*| = {len(X_star)}   截断 = {truncated}")
    if truncated:
        print(f"[WARN] 已枚举 MAX_ENUM={MAX_ENUM} 个最优解仍未穷尽 X*；"
              f"以下结果是被截断子集上的最优。")

    # 全员可观自检（每个枚举解都应通过原 verify）
    for t, x_val in enumerate(X_star):
        ok, _, _ = verify(x_val, N, supers)
        assert ok, f"枚举解 #{t} 未通过 verify：{x_val}"

    # ---- 阶段 II：鲁棒性评分 ----
    scored = evaluate_robustness(X_star, N, supers)

    # ---- 阶段 III：筛 X** ----
    R_best, X_double = select_robust_optimal(scored)
    print(f"\n[ROBUST] R_best (max R_min) = {R_best} / {n}")
    print(f"[ROBUST] |X**| (并列鲁棒最优数) = {len(X_double)}")
    if len(X_double) > 50:
        print(f"[INFO] 并列鲁棒最优解较多 (={len(X_double)})，全部输出。")

    # ---- 输出每个 X** 解的细节 ----
    print()
    for idx, s in enumerate(X_double, 1):
        print(f"[SOLUTION {idx}/{len(X_double)}]")
        print(f"    PMU = {s['P']}")
        print(f"    R_min = {s['R_min']}    R_avg = {s['R_avg']:.4f}")
        print(f"    Surv  = {s['surv']}    (按 PMU 编号升序)")
        for p, unobs in s["weakest"]:
            print(f"    最脆弱: PMU={p} 失效 → 不可观节点 = {unobs}")
        print()

    # ---- 二级辅助：R_avg 排名前 5 的最优解（仅信息，不影响决策）----
    if len(X_double) > 1:
        ranked = sorted(X_double, key=lambda s: -s["R_avg"])[:5]
        print("[AUX] X** 内按 R_avg 降序前 5：")
        for s in ranked:
            print(f"    R_avg={s['R_avg']:.4f}  R_min={s['R_min']}  PMU={s['P']}")


if __name__ == "__main__":
    main()
    # 镜像写入文本文件，避免控制台编码问题
    with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
        with contextlib.redirect_stdout(f):
            main()
