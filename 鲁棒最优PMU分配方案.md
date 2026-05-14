# 鲁棒最优 PMU 分配方案（单 PMU 失效，最坏情形最优）

> 本方案在现有 `pmu_placement.py`（最少 PMU = 10）基础上扩展。
> 目标：在所有 **PMU 数 = k\* = 10** 的最优解中，挑出**任一 PMU 失效后仍可观节点数的最坏值最大**的那个/那些方案。
>
> 编码工作严格在用户对本方案二次确认后才启动。

---

## 1. 用户决议（已确认）

| 编号 | 议题 | 决议 |
|---|---|---|
| 1 | 鲁棒性度量 | **A. max-min（最坏情形）** |
| 2 | 失效台数 | **恰好 1 台** |
| 3 | 失效后 ZI 推断 | **默认仍生效**（拓扑、ZI 节点集均不变，仅 PMU 集减一） |
| 4 | 并列最优 | **全部输出** |
| 5 | 枚举规模 | **截断 + 警告**（设 `MAX_ENUM`，超出时停枚举并打印告警） |
| 6 | 实现路线 | **枚举 + 筛选**（先枚举 `X*`，再二阶段评分挑最优） |

---

## 2. 数学模型

### 2.1 符号定义（与 `pmu_placement.py` 一致）

| 符号 | 含义 | 维度/类型 |
|---|---|---|
| `n = 37` | 节点总数 | 标量 |
| `V = {1,…,n}` | 节点集 | 集合 |
| `N[i]` | 节点 i 的闭邻域（含自环 `A_ii = 1`） | 集合 |
| `Z` | ZI 节点集 | 集合 |
| `(C, S_C)` | ZI 连通分量及其超群（`k=|C|`, `m=|S_C|`） | 见现行代码 |
| `x ∈ {0,1}^n` | PMU 放置向量；`x_i = 1` ⇔ 节点 i 装 PMU | 二进制向量 |
| `P(x) = {i : x_i = 1}` | 已装 PMU 节点集 | 集合 |
| `Obs(P)` | 在 PMU 集 `P` 下，闭邻域直接可观 + ZI 规则迭代到不动点的可观节点集 | 集合 |
| `k* = 10` | 最少 PMU 数（已由现行 ILP 求出） | 标量 |
| `X*` | 所有最优解的集合 | `X* ⊂ {0,1}^n` |

### 2.2 可观函数 `Obs(P)`（Option B：在超图 G' 上传播）

ZI 处理与 `pmu_placement.py` 同步采用 Option B（拓扑折叠为单一超节点）：

**构造 G'**
- $V' = (V\setminus Z)\cup\{v_1,\dots,v_P\}$，每个 ZI 连通分量 $C_p$ 折叠为 $v_p$
- 边按 §算法方案.md §三-B 步骤 1 重连；$v_p$ 含自环

**决策提升**
$$
x'_w = x_w\;\;(w\in V\setminus Z),\qquad
x'_{v_p}=\bigvee_{z\in C_p} x_z
$$

**G' 上的不动点伪代码**

```
P' ← {w ∈ V∖Z : x_w=1} ∪ {v_p : ∃z∈C_p, x_z=1}
Obs' ← { v ∈ V' : N'[v] ∩ P' ≠ ∅ }            # G' 中直接可观
repeat
    for each v_p ∈ Z':                          # 单 ZI 规则 (k=1)
        if |N'[v_p] ∩ Obs'| ≥ m_p - k_p:        # |N'[v_p]| = m_p-k_p+1
            Obs' ← Obs' ∪ N'[v_p]
until Obs' 不再变化

# 映回 V
Obs ← { i ∈ V∖Z : i ∈ Obs' } ∪ ⋃_{p: v_p∈Obs'} C_p
return Obs
```

> 与 §三-A 的不同已在 算法方案.md §三-B 步骤 5 表格中说明：本问题里只在
> $C=\{8,9,18\}$ 与 $C=\{24,25,28\}$ 两个 $k\ge2$ 分量上有真实差异。

### 2.3 阶段 I — 枚举所有最优解 `X*`

$$
X^* = \Big\{\, x \in \{0,1\}^n \ \Big|\ \text{ZI-SOP 约束}(x) = 1,\ \sum_{i=1}^n x_i = k^* \Big\}
$$

**枚举手段（no-good cut）**：
1. 在现行 ILP 上**追加等式约束** `Σ x_i = k*`，将原 `min Σ x_i` 改为 **可行性求解**（目标常数即可）。
2. 求解一次，得到 `x^(1)`，记 `P^(1) = P(x^(1))`。
3. 加入剪枝：
   $$
   \sum_{i \in P^{(t)}} x_i \le k^* - 1
   $$
   该约束的语义：禁止再次出现完全相同的 PMU 集合（因 `Σ x_i = k*` 已固定，等价于"至少换掉一个 PMU"）。
4. 重新求解，得 `x^(t+1)`；若 MILP 不可行 → 枚举完毕；若 `|X*|` 累计达到 `MAX_ENUM` → **截断并打印警告**。

**正确性说明**：在 `Σ x_i = k*` 约束下，no-good cut `Σ_{i∈P} x_i ≤ k*−1` 当且仅当排除"PMU 集与 P 完全相同"的解；不会误剔其他解。

### 2.4 阶段 II — 鲁棒性评分与筛选

对每个 `x ∈ X*`，记 `P = P(x)`，对每个 `p ∈ P` 定义：
$$
\text{Surv}(x, p) \;=\; \big| \text{Obs}(P \setminus \{p\}) \big|
$$

最坏情形鲁棒性度量：
$$
R_{\min}(x) \;=\; \min_{p \in P}\, \text{Surv}(x, p)
$$

最优集：
$$
X^{\star\star} \;=\; \arg\max_{x \in X^*} R_{\min}(x)
$$

按议题 4，`X**` 内**全部解一并输出**。

### 2.5 二级附加输出（不参与决策，仅供分析）

对每个 `x ∈ X**`，同时打印：
- `R_min(x)`（最坏情形可观数）
- `R_avg(x) = (1/k*) Σ_p Surv(x, p)`（平均情形可观数）
- 失效后**最脆弱**的 PMU 列表 `argmin_p Surv(x, p)` 与对应的不可观节点集
- 每台 PMU 失效后的 `Surv` 完整向量（长度 = `k*`）

---

## 3. 约束与边界条件

- `X*` 中每个 `x` 必须严格满足现行 SOP 约束（与 `solve_ilp` 等价）。
- 失效计算时拓扑 `N[·]`、ZI 节点集 `Z`、超群 `(C, S_C)` 均**保持不变**。
- `MAX_ENUM` 取值建议：默认 `5000`。可由命令行/常量调节。
- 若 `MAX_ENUM` 触发截断：输出告警 `"[WARN] 已枚举 MAX_ENUM 个最优解仍未穷尽 X*，结果是被截断子集上的最优。"`
- 若 `X**` 仍含大量并列解（如 > 50），全部输出但额外打印 `"[INFO] 共 N 个并列鲁棒最优解"`。

---

## 4. 算法步骤（伪代码）

```
INPUT : 同现行项目（EDGES, ZI_NODES）
PARAM : MAX_ENUM = 5000

# 阶段 0：复用现行流程获得 k*
n, N         = build_neighbors(EDGES)
constraints, supers = build_constraints(n, N, ZI_NODES, EDGES)
k_star, _, _, _    = solve_ilp(n, constraints)         # = 10

# 阶段 I：枚举 X*
X_star = []
add_equality(Σ x_i == k_star)                          # 新增等式约束
loop:
    if len(X_star) >= MAX_ENUM:
        warn("MAX_ENUM 截断"); break
    res = solve_milp(...)
    if not feasible: break
    x_t = res.x
    X_star.append(x_t)
    add_cut(Σ_{i ∈ P(x_t)} x_i ≤ k_star - 1)

# 阶段 II：评分
scores = []
for x in X_star:
    P = P(x)
    surv = [|Obs(P \ {p})| for p in P]
    R_min = min(surv);  R_avg = mean(surv)
    scores.append((R_min, R_avg, surv, P))

# 阶段 III：筛选 X**
R_best   = max(s.R_min for s in scores)
X_double = [s for s in scores if s.R_min == R_best]

# 输出：|X*|, |X**|, 每个 x ∈ X** 的详细鲁棒性分解
```

---

## 5. 复杂度与风险评估

| 风险 | 描述 | 缓解 |
|---|---|---|
| `|X*|` 爆炸 | 37 节点配 10 PMU 在 ZI 退化下并列解可能极多 | `MAX_ENUM` 截断+警告（议题 5 已定） |
| 重复求解 ILP | 每次 cut 后重新求解可能慢 | 可复用 `LinearConstraint` 的稀疏结构动态追加行；`scipy.milp` 不支持热启动，必要时改 `pulp + CBC` 或 `mip` |
| `Obs(P\{p})` 重算开销 | 每个 `x` 需 `k* = 10` 次评估 | 评估本身 O(n·迭代次数) 极快，无需优化 |
| 数值边界 | `x` 由 ILP 返回浮点；需 `> 0.5` 阈值化 | 与现行 `solve_ilp` 一致 |
| 枚举不完备风险 | no-good cut 写错可能误剔 | 单元测试：cut 加入后重解必不返回 `P^(t)`；并验证 `Σ x_i = k*` 始终成立 |

---

## 6. 输出格式约定

控制台 + `result_robust.txt` 双写（沿用现行风格）。建议格式：

```
[INFO] k* = 10  |X*| = <num>  (是否截断: yes/no)
[INFO] R_best (max-min Surv) = <value>
[INFO] |X**| (并列鲁棒最优数) = <num>

[SOLUTION 1/<N>] PMU = [...]
                 R_min = ..,  R_avg = ..
                 Surv 向量 = [..]   (按 PMU 编号升序)
                 最脆弱 PMU = [..], 失效后不可观节点 = [..]

[SOLUTION 2/<N>] ...
```

---

## 7. 隐式假设备案

1. ZI 推断规则在 PMU 子集上仍然成立（已通读 `verify` 确认无"PMU 数 = 10"硬编码）。
2. `MAX_ENUM = 5000` 是工程默认值，非数学硬性约束；如不合适可调。
3. 复用现行 `build_constraints` 与 `verify` 的 SOP/ZI 实现，不修改其逻辑。新增模块只追加约束、循环求解、二阶段评分。
4. 假设 ILP 求解器在每次 cut 后仍能正确返回最优/不可行；不依赖热启动。

---

## 8. 待用户二次确认

请回复以下任一：

- **「方案确认」** → 我进入第四步（编码实现）。
- **「需调整」** → 请指出要修改的小节（例如把 `MAX_ENUM` 改为某值、追加 tiebreak 规则等）。

> 严格遵循 CLAUDE.md：未收到确认前不写任何代码。
