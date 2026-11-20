# MovieLens 推荐系统（从零搭建）

一条完整的推荐系统链路：**时序切分 → 多路召回 → 精排 → MMR 重排 → 离线评估 → FastAPI 服务 + 网页**。
不使用现成的推荐库一键完成，每一层都自己实现、可插拔、可对比。

> 当前状态：**阶段 0/1/2/3/4/6/7/8 已完成**。链路为
> `ItemCF + BPR-MF + 双塔向量召回（加权 RRF，热门仅兜底）→ MMR 重排 → 评估 → FastAPI + 网页`，
> 47 个 pytest 全过，test 集 `Recall@10 = 0.0836`，是热门基线的 2.22 倍、单路 ItemCF 的 1.28 倍。
> DIN / DeepFM 精排已定义接口与实现计划，代码为占位（见下方 Roadmap）。
> 每完成一层只需把 `implemented = True` 打开，`pipeline.py` 会自动接入，无需改动其他代码。

## 快速开始

```bash
./scripts/run_all.sh            # venv -> 下载 -> 切分 -> pytest -> 离线评估 -> FAISS 对照
./scripts/run_all.sh --serve    # 上面全部执行完再启动服务
```

或分步：

```bash
./scripts/setup.sh                                   # 建 venv + 装依赖
.venv/bin/python -m movielens.data.download         # 下载 ml-latest-small
.venv/bin/python -m movielens.data.prepare          # 时序切分 + 负采样
.venv/bin/python -m pytest                          # 47 项测试
.venv/bin/python scripts/tune_itemcf.py             # ItemCF 超参扫描（只用 valid）
.venv/bin/python scripts/tune_bpr.py                # BPR-MF 超参扫描（只用 valid）
.venv/bin/python scripts/tune_two_tower.py          # 双塔温度/历史长度扫描（只用 valid）
.venv/bin/python scripts/evaluate.py                # 消融对比 + MMR λ 扫描
.venv/bin/python scripts/compare_index.py           # Flat/HNSW/NumPy 精确检索对照
.venv/bin/python -m uvicorn movielens.api.main:app --port 8000
# 打开 http://127.0.0.1:8000/
```

> 沙箱/共享 tmp 环境下若 pytest 报 `FileExistsError: ... pytest-of-root`，是临时目录残留导致，
> 与代码无关：`.venv/bin/python -m pytest --basetemp=/tmp/ml-pytest-$(date +%s)` 即可；`run_all.sh` 已自动使用独立 basetemp。

接口示例：`GET /recommend/1?k=10&lam=0.7`，另有 `/health`、`/users`、`/movies/{movie_id}`、`/metrics`。
换数据集：改 `configs/default.yaml` 的 `data.dataset` 为 `ml-1m`（该数据集含用户人口属性，双塔会自动编码进用户塔），
切分方式可切 `split.strategy: leave_last_out`。

## 环境说明

- 使用**标准 venv**（`.venv/`，Python 3.12）。本机只有 3.13，所以用 `uv python install 3.12` 拉解释器，
  避开 `faiss-cpu` / `torch` 对新 Python wheel 的滞后；没有 uv 时脚本自动退回 `python3 -m venv`。
- `uv` 只用于下载解释器和加速装包，运行期完全不依赖它，`requirements.txt` + `pip` 同样可复现。
- 个别沙箱环境无法构建 editable wheel，`scripts/setup.sh` 会自动降级为在 site-packages 写一个
  `.pth` 指向 `src/`，导入行为等价（已在本机验证）。
- **macOS 的 OpenMP 兼容处理**：当前 `torch` 与 `faiss-cpu` wheel 各自携带 `libomp.dylib`，同一进程
  先做 torch 计算再执行 FAISS 查询会触发 `OMP Error #15`；预加载或强制兼容开关在本机验证均不可用。
  因此 `backend: auto` 在 macOS 使用 NumPy 做**与 `IndexFlatIP` 等价的精确内积检索**（实测逐位结果一致），
  其它平台直接使用 FAISS。真正的 Flat/HNSW 索引由不含 torch 的 `faiss_index.py` 和 `compare_index.py`
  在独立进程中构建、持久化并对照，服务链路不受影响。
- 数据、模型、索引、指标文件都不入库（见 `.gitignore`），全部由脚本复现。

## 目录结构

```
configs/default.yaml          全部超参与口径（切分比例、K、λ、长尾分位…）
src/movielens/
  config.py                   类型化配置加载
  schema.py                   统一协议：Candidate / Recommendation / Metrics / 各层 ABC
  data/download.py            MovieLens 下载（csv 与 dat 两种格式）
  data/prepare.py             时序切分、id 映射、seen、负采样、genre 矩阵
  data/dataset.py             Dataset 对象（训练/评估/服务共用）
  recall/popular.py           热门基线 + 冷启动/候选不足兜底
  recall/itemcf.py            ItemCF：共现、热度惩罚、IUF、top-N 邻居、矩阵缓存
  recall/two_tower.py         双塔训练、InfoNCE、embedding 缓存、召回适配
  faiss_index.py              独立于 torch 的 Flat/HNSW 索引与显式 id 映射
  mf/bpr.py                   BPR-MF：pairwise 训练、动态负采样、valid 早停、向量缓存
  rank/{din,deepfm}.py        精排（占位）
  rerank/mmr.py               MMR 重排
  eval/metrics.py             Recall/NDCG/覆盖率/长尾/新颖度/ILD + 分数矩阵快速评估
  pipeline.py                 唯一的编排链路（离线与在线共用）
  api/main.py                 FastAPI
  web/index.html              演示网页（推荐结果 + 消融表 + 索引对照）
scripts/                      setup.sh / run_all.sh / evaluate.py / compare_index.py / tune_*.py
tests/                        指标、切分防穿越、ItemCF 手算、BPR、双塔、FAISS、召回融合、MMR、API 契约
artifacts/metrics/            离线指标快照（脚本生成）
artifacts/models/{itemcf,bpr,two_tower}/  模型与索引缓存（按 train 数据 + 超参指纹命名）
```

## 数据流与关键约定

```
全量物品池 → 个性化多路召回(加权 RRF) → 热门兜底补尾 → 精排打分 → MMR 重排 → Top-K
                  ↑ 每一步都过滤 seen，且只用时间点之前的数据
```

1. **统一候选格式**：每层都输出 `list[Candidate]`，内部一律用连续的 `user_idx / item_idx`，
   只在 API 出口映射回原始 `userId / movieId`。这是各层可插拔的前提。
2. **个性化源之间用加权 RRF 融合**（`weight/(60+rank)`），避免把 ItemCF 分、BPR 分、双塔向量分这些
   量纲完全不同的分数直接比较；**热门源默认不参与竞争**，只在主源为空（冷启动）或候选数不足
   `min_primary_candidates` 时补在候选池尾部。原因见下方消融表：等权 RRF 会让热门物品把
   个性化候选挤掉，覆盖率从 0.0946 塌到 0.0304。
3. **时间切分不随机**：默认 `leave_last_out`（每个用户最后一条 = test、倒数第二条 = valid），
   这样 610 个用户都有测试正例、模型对比更稳；`configs/default.yaml` 可切 `global`
   （按全局时间戳 80/10/10 切，更贴近线上，但 test 用户少、冷启动占比高）。
   两种策略都有硬断言：任何 `(user, item)` 对不得跨 split 出现。
4. **seen 分场景**：评估 valid 时 seen 只含 train；评估 test 时 seen 含 train+valid；
   线上服务用 `seen["all"]` 过滤用户全部历史。冷启动用户走热门榜兜底。
5. **负采样**：只从 train 时间窗内出现过的电影里采，按 `hard_negative_ratio` 混合均匀负例与
   热度负例，且用布尔矩阵向量化地剔除用户已看过的物品，绝不把未来新片当负例。
6. **离线/在线同一份 `pipeline.py`**，避免训练与服务不一致（train/serve skew）。

## 指标口径（README 写清楚，面试才讲得清）

| 指标 | 定义 |
| --- | --- |
| Recall@10 | `|Top10 ∩ test 正例| / |test 正例|`，按用户平均 |
| NDCG@10 | 二值相关性 gain=1，`DCG / IDCG`，IDCG 由该用户全部正例构造；也支持 `2^rel-1` 分级 gain |
| Coverage | 所有用户 Top10 的**去重物品数 / 全量电影数** |
| Long-tail coverage | 被推荐到的长尾物品数 / 长尾物品总数；长尾 = train 交互数 ≤ `quantile(tail_quantile)`（默认 0.8，即流行度最低的 80% 电影） |
| Long-tail share | 推荐位中属于长尾物品的比例 |
| Novelty | 平均 `-log2(物品流行度占比)`（带平滑） |
| ILD | 每个用户 Top10 内平均成对 `1 - genre 余弦相似度` |

所有模型共用同一个 `evaluate.py`、同一个 test 集、同一个 K、同一套 seen 过滤，保证对比公平。

### 消融对比（ml-latest-small，leave_last_out，610 个 test 用户，test 集）

| 方案 | Recall@10 | NDCG@10 | Coverage | 长尾覆盖 | 长尾占比 | ILD |
| --- | --- | --- | --- | --- | --- | --- |
| popular | 0.0377 | 0.0189 | 0.0124 | 0.0000 | 0.0000 | 0.6987 |
| popular + MMR | 0.0426 | 0.0199 | 0.0125 | 0.0000 | 0.0000 | 0.7473 |
| itemcf | 0.0607 | 0.0297 | 0.0935 | 0.0523 | 0.0926 | 0.6685 |
| itemcf + MMR | 0.0656 | 0.0316 | 0.0945 | 0.0536 | 0.0964 | 0.7180 |
| bpr | 0.0590 | 0.0302 | 0.1225 | 0.0131 | 0.0200 | 0.6713 |
| bpr + MMR | 0.0607 | 0.0311 | 0.1204 | 0.0132 | 0.0195 | 0.7173 |
| two_tower | 0.0639 | 0.0334 | 0.3172 | 0.2228 | 0.4159 | 0.6057 |
| two_tower + MMR | 0.0689 | 0.0345 | 0.3169 | 0.2232 | 0.4238 | 0.6778 |
| itemcf + bpr + MMR（上一阶段默认） | 0.0820 | 0.0401 | 0.0894 | 0.0187 | 0.0280 | 0.7298 |
| **itemcf + bpr + two_tower + 热门兜底 + MMR（默认）** | **0.0836** | **0.0414** | **0.1394** | **0.0486** | **0.0770** | **0.7179** |
| itemcf + bpr + two_tower + MMR | 0.0836 | 0.0414 | 0.1394 | 0.0486 | 0.0770 | 0.7179 |
| itemcf + 热门兜底 + MMR | 0.0656 | 0.0316 | 0.0946 | 0.0536 | 0.0964 | 0.7188 |
| popular + itemcf 等权 RRF + MMR | 0.0574 | 0.0260 | 0.0304 | 0.0095 | 0.0152 | 0.7339 |

五个值得记住的结论：

1. **协同过滤相对热门基线**：Recall `0.0377 → 0.0656`（+74%），Coverage `0.0124 → 0.0946`（7.6 倍），
   长尾占比 `0 → 0.0964`。热门基线长尾覆盖恒为 0，就是马太效应的量化证据。
2. **多路个性化召回互补**：ItemCF + BPR 两路 RRF 后 Recall `0.0820`，比单路分别高 25% / 35%。
   BPR 隐向量覆盖更广、ItemCF 共现图更偏小众；加入双塔后 Recall 再升到 `0.0836`。
3. **双塔主要贡献不是命中率，而是曝光结构**：单路双塔 Recall 只有 0.0639，但 Coverage 0.3172、
   长尾占比 0.4159，远高于 ItemCF/BPR。三路融合把覆盖率从 0.0894 抬到 0.1394（+56%）、
   长尾占比从 0.0280 抬到 0.0770（2.75 倍），NDCG 同步提升，ILD 仅从 0.7298 小幅降到 0.7179。
4. **MMR 不只是"牺牲相关性换多样性"**：ItemCF 单路加 MMR 后 Recall 从 0.0607 升到 0.0656、ILD 从
   0.6685 升到 0.7188；双塔单路加 MMR 后 Recall 从 0.0639 升到 0.0689。原始 Top10 常挤满同一批
   高相似物品，去同质化后正确答案才有机会进入前 10。
5. **多路召回不是"加一路就更好"**：把热门和 ItemCF 等权 RRF 后 Recall 降到 0.0574、长尾占比跌到
   0.0152。热门物品永远占据 popular 路的 rank 0，RRF 给它们稳定的 `1/61` 分，个性化候选反而被稀释。
   RRF 权重扫描（valid）也显示 0.005 以内的差异只有一个测试用户的量级，默认保持等权。

复现：`scripts/evaluate.py`，结果写入 `artifacts/metrics/{ablation,baseline,mmr_tradeoff}.json`（不入库）。

## ItemCF 实现要点

```
R = train 隐式矩阵(IUF 加权)     C = RᵀR     sim = C_ij / (|N(i)|^α·|N(j)|^α + shrink)
s(u,i) = Σ_{j ∈ 用户 train 历史} sim(j,i) · exp(-decay · age_days(j))
```

- **共现只在 train 上算**，`min_cooccurrence` 过滤用的是**未加权**的共现次数——IUF 后权重 <1，
  若对加权值设阈值会误删绝大多数真实共现对（这个坑在测试 `test_min_cooccurrence_uses_raw_support` 里锁住）。
- **top-N 邻居**：9742×9742 全量相似度既没必要也存不下有效信息，截断后每个物品只留最相似的 20 个，
  平均 19.9 个邻居、60 个物品无邻居（训练集里只出现一次的长尾片）。相似度矩阵按
  `blake2b(train 数据 + 超参)` 指纹缓存到 `artifacts/models/itemcf/`，服务启动命中缓存即跳过拟合。
- **用户画像只用 train 历史**，最近 `history_size=100` 条；用 valid/test 物品当画像等于把答案喂给召回。
- **推荐理由可溯源**：`explain()` 返回贡献最大的历史物品，API 输出「与你喜欢的《X》常被一起观看」。
- 冷启动/无邻居用户返回空列表，由热门兜底接管，不会报错或推已看。

超参在 **valid** 上扫描（`scripts/tune_itemcf.py`，绝不碰 test），结论：

| 超参 | 现象 | 选择 |
| --- | --- | --- |
| `alpha` | 0.5 最优；α=1 时 Recall 崩到 0.0033、长尾占比 0.9987（惩罚过度→只剩长尾，相关性崩） | 0.5 |
| `iuf` | 开启 Recall 0.0590 vs 关闭 0.0492，重度用户不再制造虚假共现 | true |
| `shrink` | 增大持续降低 Recall 与覆盖率，本数据集低支持噪声不是主要矛盾 | 0.0 |
| `top_n_neighbours` | 20 与 80 的 Recall 相同，但 20 的 NDCG/覆盖率/长尾占比更高 | 20 |
| `history_decay` | 0 → 0.05 时 Recall 0.0607 → 0.0672、NDCG 0.0292 → 0.0350；0.05~0.2 之间平稳 | 0.05 |

## BPR-MF 实现要点

```
x_ui = p_u · q_i        L = -ln σ(x_ui - x_uj) + λ(||p_u||² + ||q_i||² + ||q_j||²)
```

- **为什么需要它**：ItemCF 只能记忆显式共现，训练集里出现次数太少的电影没有任何邻居（本数据集 60 部），
  MF 把每个用户/物品放进同一个隐向量空间，能对"从未共现过"的物品对打分；BPR 直接优化**排序**
  （正例必须排在采样的负例之前），而不是还原评分，和 Top-K 目标一致。
- **负例每个 epoch 重采**，不是固定一张 pair 表：固定负例几轮就被学穿。采样用布尔 seen 矩阵向量化拒绝，
  只从 train 窗口内出现过的物品里采，支持均匀 + 热度（hard）混合，比例可调。
- **早停用 valid 的 Recall@10**，每次评估只做一次 `user_factors @ item_factors.T` 稠密打分（610×9742），
  不走完整 pipeline，因此 5 个 epoch 评估一次也几乎没有成本；训练结束恢复 best checkpoint。
- **过拟合曲线很典型**：train loss 从 0.514 一路降到 0.015，valid Recall 却在 epoch 15 达峰（0.0639）
  后一路跌到 0.0393。**pairwise loss 下降 ≠ 排序质量上升**，早停不是可选项。
- `user_factors / item_factors` 已导出，双塔可以扩展为用它们初始化；`score(u,i)` 也可作为精排特征。
- 因子按 `blake2b(train 正例 + 训练超参)` 指纹缓存到 `artifacts/models/bpr/`，改任何超参都会触发重训。

超参在 **valid** 上扫描（`scripts/tune_bpr.py`），结论与一个重要提醒：

| 超参 | 现象 | 选择 |
| --- | --- | --- |
| `dim` | 32 最优 0.0672；16 欠拟合 0.0607，64/128 过拟合提前 | 32 |
| `reg` | 0.0001→0.01 时 Recall 0.0639→0.0705，正则越强越好，印证过拟合是主要矛盾 | 0.001（见下） |
| `hard_negative_ratio` | 纯均匀 0.0705 > 混合 0.0639 > 全热度 0.0656，本数据集热度负例无收益 | 0.0 |
| `negatives_per_positive` | 1 / 4 / 8 → 0.0656 / 0.0639 / 0.0689，收益不抵训练时长 | 4 |
| `lr` | 0.001 / 0.005 / 0.01 → 0.0672 / 0.0639 / 0.0639 | 0.01 |

**单变量最优不可叠加**：`reg=0.01 + 均匀负例` 组合后 Recall 反而只有 0.0607。原因是 leave-last-out 下
每个用户只有 1 个 valid 正例，`1/610 ≈ 0.0016` 就是一个用户的权重，0.005 以内的差异基本是噪声。
所以最终只选一组稳定配置（dim=32、reg=0.001、均匀负例），不继续在小验证集上"刷点"。

## 双塔 + FAISS 实现要点

```
用户塔 v_u = L2(MLP([user_id_emb, mean(history item_id_emb), demographics]))
物品塔 v_i = L2(MLP([item_id_emb, genre multi-hot, normalised year]))
InfoNCE = -log exp(v_u·v_i / τ) / Σ_{j ∈ in-batch + 额外负例, j 非 seen} exp(v_u·v_j / τ)
```

- **训练历史是严格前缀**：对 train 中时间位置 t 的正例，用户画像只含 t 之前的交互；评估/服务才用完整
  train 历史。这样既没有未来穿越，也避免把当前正例直接塞进用户画像造成捷径学习。
- **用户/物品塔只在最后内积交互**，物品向量可 T+1 全量离线导出；两塔输出 L2 归一化，内积即 cosine。
  `ml-1m` 的 gender / age / occupation 会作为静态 one-hot 特征自动接入用户塔。
- **负例 = in-batch positives + 每个正例 32 个额外负例**，额外负例混合均匀与 `popularity^0.75` 采样，
  只从 train 窗口内出现过的物品里采。用户已看过的 batch 内正例和额外负例都被 explicit false-negative
  mask 置为 `-1e4`，对角线本身例外。测试用固定张量手算 mask 前后 loss，锁住这一行为。
- **early stopping 用全链路无关的稠密 valid Recall@10**：每 5 epoch 计算一次用户向量 × 全量物品向量，
  恢复 best checkpoint；test 从不参与训练。完整训练约 133s，best epoch 30。
- **索引 id 映射**：`IndexIDMap2` 显式写入 `np.arange(n_items)`，FAISS 返回的 id 永远等于连续 `item_idx`；
  Flat 与 HNSW 都按 `blake2b(train 数据 + 超参)` 指纹持久化，HNSW 单独设置 `efConstruction / efSearch`。
- **精确 / 近似 / 回退对照**（`scripts/compare_index.py`，同一份最终 embedding）：

| 检索方式 | Recall@10 | NDCG@10 | 平均延迟 | P95 延迟 |
| --- | --- | --- | --- | --- |
| FAISS `IndexFlatIP` | 0.0639 | 0.0334 | 0.183 ms | 0.240 ms |
| FAISS `IndexHNSWFlat` | 0.0639 | 0.0334 | 0.142 ms | 0.202 ms |
| NumPy 精确回退 | 与 Flat Top10 重合度 100% | 同左 | macOS 在线默认 | — |

Flat/HNSW 的 Top10 平均重合度 **97.93%**、候选集@200 重合度 **80.29%**；NumPy 回退与 Flat
的 @200 列表逐位一致率 **100%**。HNSW 延迟更低但会改变候选池尾部，这正是 ANN 的召回率/延迟取舍。

**调参必须对齐最终目标**（`scripts/tune_two_tower.py`，仅 valid）：单模型看 `τ=0.2` 最优
（Recall 0.1000 / NDCG 0.0485，`τ=0.1` 是 0.0902 / 0.0431），但把它接进三路 RRF 后全链路 valid
降到 0.0885 / 0.0444；`τ=0.1` 的全链路 valid 是 0.1033 / 0.0485、test 是 0.0836 / 0.0414。
`history_size=64` 单模型也比 32 略好，但训练更慢、链路无增益。因此默认保持 `τ=0.1 / history_size=32`——
**单塔离线最优不等于多路融合最优**，这是比多报 0.01 更值得讲清的结论。

## Roadmap

| 阶段 | 内容 | 状态 |
| --- | --- | --- |
| 0 | venv、配置、目录、统一协议、脚本 | 已完成 |
| 1 | 下载、时序切分、防穿越断言、负采样 | 已完成 |
| 2 | 热门基线（含时间衰减） | 已完成 |
| 2 | ItemCF：稀疏共现 + 热度惩罚 + IUF + top-N 邻居 + 矩阵缓存 | 已完成（含 valid 调参） |
| 3 | BPR-MF：pairwise loss、动态负采样、valid 早停、因子缓存 | 已完成（含 valid 调参） |
| 4 | 双塔 + FAISS：in-batch negatives、false-negative mask、温度系数、Flat/HNSW 对照 | 已完成（含 valid 调参与索引对照） |
| 5 | DIN（行为序列 attention）/ DeepFM 对照，上游分数作为特征 | 接口占位 |
| 6 | MMR 重排 + λ 扫描的相关性/多样性权衡曲线 | 已完成 |
| 7 | Recall@10、NDCG@10、覆盖率、长尾覆盖率 + 消融对比 | 已完成（13 组消融） |
| 8 | FastAPI `/recommend/{user_id}` + 网页 + 冷启动兜底 + 分步耗时 | 已完成 |

每完成一层，在对应类里把 `implemented` 改为 `True` 并实现 `recall` / `predict` 即可自动接入
`RECALL_REGISTRY` / `RANK_REGISTRY`，评估与 API 无需修改。

## 面试主线

每一层要能说清「解决了上一层什么问题、代价是什么」：热门无个性化且有马太效应 → ItemCF 引入协同
但稀疏/冷启动差 → BPR 用隐向量泛化并直接优化排序 → 双塔加入 side feature 且能用 ANN 检索百万候选，
但用户与物品直到最后才交互 → 精排做交叉特征与兴趣建模，只能打几百个候选 → 重排修多样性与长尾 →
评估暴露离线指标与线上目标的错位 → 服务暴露延迟、缓存、降级、embedding 版本一致性。
高频追问：为什么必须时间切分、负采样与 false-negative mask、曝光偏差与样本选择偏差（SSB）、
in-batch negatives 和温度系数、精确/近似检索、冷启动、A/B 与线上指标、多路召回融合
（本项目实测：热门参与等权 RRF 会伤害覆盖率与长尾；三路个性化召回使 Recall 0.0820 → 0.0836、
覆盖率 0.0894 → 0.1394）、单模型指标与全链路指标错位、BPR 的负采样与早停、MMR 为何有时提升 Recall。

数据源：MovieLens（GroupLens），`ml-latest-small` / `ml-1m`，仅用于学习与研究。
