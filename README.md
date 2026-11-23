# MovieLens 推荐系统

从零实现的一条推荐链路：时序切分 → 多路召回 → 可选精排 → MMR 重排 → 离线评估 → FastAPI + 网页。各层可插拔，离线评估与线上服务共用同一条 pipeline。

当前默认方案为 ItemCF + BPR-MF + 双塔（加权 RRF 融合，热门仅兜底）→ MMR，test 集 `Recall@10 = 0.0836`、`NDCG@10 = 0.0414`。DeepFM 和 DIN 已接入并完成对照，默认设置依据 valid 保持 `rank.model: none`。

[项目说明](docs/项目说明.md)从一次真实推荐请求讲起，介绍数据、模型和服务架构，以及每一层对应的搜广推技术。

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
.venv/bin/python -m pytest                          # 50 项测试
.venv/bin/python scripts/tune_itemcf.py             # ItemCF 超参扫描（只用 valid）
.venv/bin/python scripts/tune_bpr.py                # BPR-MF 超参扫描（只用 valid）
.venv/bin/python scripts/tune_two_tower.py          # 双塔温度/历史长度扫描（只用 valid）
.venv/bin/python scripts/evaluate.py                # 消融对比 + MMR λ 扫描
.venv/bin/python scripts/train_rank.py              # 训练 DeepFM / DIN，并缓存模型
.venv/bin/python scripts/evaluate.py --rank         # 加入精排对照，重新扫描 MMR λ
.venv/bin/python scripts/compare_index.py           # Flat/HNSW/NumPy 精确检索对照
.venv/bin/python -m uvicorn movielens.api.main:app --port 8000
# 打开 http://127.0.0.1:8000/
```

> 沙箱/共享 tmp 环境下若 pytest 报 `FileExistsError: ... pytest-of-root`，是临时目录残留导致，
> 与代码无关：`.venv/bin/python -m pytest --basetemp=/tmp/ml-pytest-$(date +%s)` 即可；`run_all.sh` 已自动使用独立 basetemp。

接口示例：`GET /recommend/1?k=10&lam=0.7`，另有 `/health`、`/users`、`/movies/{movie_id}`、`/metrics`。
换数据集：改 `configs/default.yaml` 的 `data.dataset` 为 `ml-1m`（该数据集含用户人口属性，双塔会自动编码进用户塔），
全局时间切分可用 `split.strategy: global`。

## 部署演练

先生成数据、模型缓存和指标。打包脚本只携带当前配置使用的模型、处理数据、指标和服务源码。

```bash
uv pip compile --python .venv/bin/python requirements.txt -o /tmp/movielens-runtime.txt
.venv/bin/python scripts/package_release.py dist/movielens-release --lock /tmp/movielens-runtime.txt
cd dist/movielens-release
uv venv --python 3.12 .venv
uv pip sync --python .venv/bin/python requirements.lock.txt
.venv/bin/python scripts/smoke_deploy.py
```

锁文件按目标系统和架构解析。演练脚本只在 `127.0.0.1` 的临时端口启动 Uvicorn，用真实 HTTP 请求检查首页、就绪状态、普通推荐、冷启动、电影详情和指标。重启后对比推荐结果，并检查模型缓存文件是否被改写。Linux 上也使用同一脚本验收 FAISS 服务。

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
  rank/                       候选采样、共享特征、DeepFM / DIN 及模型缓存
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
                  ↑ 召回时过滤 seen，精排使用训练窗快照特征
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
   两种策略均断言 `(user, item)` 对不会跨 split。按用户留出可能让共享模型看到晚于某个用户评估事件的其他用户行为。
4. **seen 分场景**：评估 valid 时 seen 只含 train；评估 test 时 seen 含 train+valid；
   线上服务用 `seen["all"]` 过滤用户全部历史。冷启动用户走热门榜兜底。
5. **负采样**：只从 train 时间窗内出现过的电影里采，按 `hard_negative_ratio` 混合均匀负例与
   热度负例，且用布尔矩阵向量化地剔除用户已看过的物品，绝不把未来新片当负例。
6. **离线/在线同一份 `pipeline.py`**，避免训练与服务不一致（train/serve skew）。

## 指标口径

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

结果要点：

- ItemCF + MMR 相比热门基线：Recall `0.0377 → 0.0656`，Coverage `0.0124 → 0.0946`，长尾占比 `0 → 0.0964`；热门基线长尾覆盖恒为 0。
- ItemCF + BPR 两路 RRF 的 Recall 为 `0.0820`，高于单路最佳（`itemcf + MMR 0.0656`、`bpr + MMR 0.0607`）；加入双塔后 `0.0836`。
- 双塔单路 Recall 仅 0.0639，但 Coverage 0.3172、长尾占比 0.4159；三路融合后覆盖率 `0.0894 → 0.1394`、长尾占比 `0.0280 → 0.0770`。
- MMR 在本数据集同时提升 Recall 与 ILD：ItemCF 单路 `0.0607 → 0.0656`、ILD `0.6685 → 0.7188`；双塔单路 Recall `0.0639 → 0.0689`。
- 热门与个性化源等权 RRF 后 Recall 降到 0.0574、长尾占比 0.0152，因此热门只作兜底；valid 上的 RRF 权重扫描差异在 0.005 以内，默认等权。

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
`history_size=64` 单模型也比 32 略好，但训练更慢、链路无增益。因此默认保持 `τ=0.1 / history_size=32`。单塔离线最优不等于多路融合最优。

## 阶段 5 精排

训练先按全局时间在 `train` 内取一个切点。召回模型、已看集合、行为序列和统计特征仅使用严格早于切点的训练交互。候选限于切点前已有交互的电影，用切点之后剩余训练窗中的交互标记命中。负例从召回候选内抽取，并在 pointwise BCE 中用抽样权重恢复候选频率。这是基于评分记录的代理标签，MovieLens 未记录真实曝光。

- 共用特征包括用户和电影 ID、类型、用户交互数与平均评分、训练交互热度和时间衰减热度。RRF 保留每路召回的分数，精排同时使用融合分、ItemCF / BPR / 双塔分数及缺失标记。
- DeepFM 使用共享特征嵌入的 FM 二阶交叉与 MLP。DIN 通过候选物品对最近历史计算注意力权重，屏蔽填充位置。
- `train_rank.py` 按 valid 全链路 Recall@10、NDCG@10 选择 checkpoint，并记录候选级 AUC 与 LogLoss。两个模型共用同一个训练切点。模型缓存随训练与验证数据、特征和配置变化失效。
- 运行 `scripts/train_rank.py` 后可将 `rank.model` 改为 `deepfm` 或 `din` 再启动 API。`/health` 显示已启用模型，推荐项显示 `rank_score`。无训练历史的用户继续走热门兜底。

在本机 `ml-latest-small` 的 `leave_last_out` 实验中，训练快照生成 18,185 条候选样本，其中 4,941 条命中。下表固定召回配置和 MMR λ=0.7，valid 和 test 各 610 位用户。

| 精排器 | Valid Recall@10 | Valid NDCG@10 | Test Recall@10 | Test NDCG@10 |
| --- | ---: | ---: | ---: | ---: |
| 关闭 | 0.1033 | 0.0485 | 0.0836 | 0.0414 |
| DeepFM | 0.0820 | 0.0361 | 0.0475 | 0.0218 |
| DIN | 0.0590 | 0.0334 | 0.0377 | 0.0192 |

`evaluate.py --rank` 输出 17 组消融，并在整个 valid 上按三种精排设置扫描 λ。无精排时 λ=1 的 valid Recall@10 为 0.1082，λ=0.7 的列表多样性 ILD 为 0.7173，高于 λ=1 的 0.6540。默认保留 λ=0.7 的相关性与多样性取舍。

## Roadmap

| 阶段 | 内容 | 状态 |
| --- | --- | --- |
| 0 | venv、配置、目录、统一协议、脚本 | 已完成 |
| 1 | 下载、时序切分、防穿越断言、负采样 | 已完成 |
| 2 | 热门基线（含时间衰减） | 已完成 |
| 2 | ItemCF：稀疏共现 + 热度惩罚 + IUF + top-N 邻居 + 矩阵缓存 | 已完成（含 valid 调参） |
| 3 | BPR-MF：pairwise loss、动态负采样、valid 早停、因子缓存 | 已完成（含 valid 调参） |
| 4 | 双塔 + FAISS：in-batch negatives、false-negative mask、温度系数、Flat/HNSW 对照 | 已完成（含 valid 调参与索引对照） |
| 5 | 时序候选采样、DeepFM / DIN 精排、上游分数作为特征 | 已完成（默认关闭，可配置启用） |
| 6 | MMR 重排 + λ 扫描的相关性/多样性权衡曲线 | 已完成 |
| 7 | Recall@10、NDCG@10、覆盖率、长尾覆盖率 + 消融对比 | 已完成（含精排共 17 组） |
| 8 | FastAPI `/recommend/{user_id}` + 网页 + 冷启动兜底 + 分步耗时 | 已完成 |

召回和精排模型分别注册到 `RECALL_REGISTRY` 与 `RANK_REGISTRY`，离线评估与 API 共用同一条推荐链路。

数据源：MovieLens（GroupLens），`ml-latest-small` / `ml-1m`，仅用于学习与研究。
