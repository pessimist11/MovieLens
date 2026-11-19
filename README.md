# MovieLens 推荐系统（从零搭建）

一条完整的推荐系统链路：**时序切分 → 多路召回 → 精排 → MMR 重排 → 离线评估 → FastAPI 服务 + 网页**。
不使用现成的推荐库一键完成，每一层都自己实现、可插拔、可对比。

> 当前状态：**基础框架 + 热门基线 + MMR + 评估 + 服务全链路已跑通**；
> ItemCF / BPR-MF / 双塔+FAISS / DIN 已定义接口与实现计划，代码为占位（见下方 Roadmap）。
> 因此现在就能端到端跑出指标，后续每完成一层只需把 `implemented = True` 打开，`pipeline.py` 会自动接入，无需改动其他代码。

## 快速开始

```bash
./scripts/run_all.sh            # venv -> 下载 -> 切分 -> pytest -> 离线评估
./scripts/run_all.sh --serve    # 上面全部执行完再启动服务
```

或分步：

```bash
./scripts/setup.sh                                   # 建 venv + 装依赖
.venv/bin/python -m movielens.data.download         # 下载 ml-latest-small
.venv/bin/python -m movielens.data.prepare          # 时序切分 + 负采样
.venv/bin/python -m pytest                          # 测试套件
.venv/bin/python scripts/evaluate.py                # 指标 + MMR λ 扫描
.venv/bin/python -m uvicorn movielens.api.main:app --port 8000
# 打开 http://127.0.0.1:8000/
```

接口示例：`GET /recommend/1?k=10&lam=0.7`，另有 `/health`、`/users`、`/movies/{movie_id}`、`/metrics`。
换数据集：改 `configs/default.yaml` 的 `data.dataset` 为 `ml-1m`（该数据集含用户人口属性，双塔阶段可用），
切分方式可切 `split.strategy: leave_last_out`。

## 环境说明

- 使用**标准 venv**（`.venv/`，Python 3.12）。本机只有 3.13，所以用 `uv python install 3.12` 拉解释器，
  避开 `faiss-cpu` / `torch` 对新 Python wheel 的滞后；没有 uv 时脚本自动退回 `python3 -m venv`。
- `uv` 只用于下载解释器和加速装包，运行期完全不依赖它，`requirements.txt` + `pip` 同样可复现。
- 个别沙箱环境无法构建 editable wheel，`scripts/setup.sh` 会自动降级为在 site-packages 写一个
  `.pth` 指向 `src/`，导入行为等价（已在本机验证）。
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
  recall/{popular,itemcf,two_tower}.py
  mf/bpr.py                   BPR-MF（占位）
  rank/{din,deepfm}.py        精排（占位）
  rerank/mmr.py               MMR 重排
  eval/metrics.py             Recall/NDCG/覆盖率/长尾/新颖度/ILD
  pipeline.py                 唯一的编排链路（离线与在线共用）
  api/main.py                 FastAPI
  web/index.html              演示网页
scripts/                      setup.sh / run_all.sh / evaluate.py
tests/                        指标、切分防穿越、MMR、API 契约
artifacts/metrics/            离线指标快照（脚本生成）
```

## 数据流与关键约定

```
全量物品池 → 多路召回(RRF 融合) → 精排打分 → MMR 重排 → Top-K
                  ↑ 每一步都过滤 seen，且只用时间点之前的数据
```

1. **统一候选格式**：每层都输出 `list[Candidate]`，内部一律用连续的 `user_idx / item_idx`，
   只在 API 出口映射回原始 `userId / movieId`。这是各层可插拔的前提。
2. **多路召回用 RRF 融合**（`1/(60+rank)`），避免把热门分、ItemCF 分、双塔向量分这些
   量纲完全不同的分数直接比较。
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

### 当前基线快照（ml-latest-small，leave_last_out，610 个 test 用户）

| 指标 | 热门召回 + MMR |
| --- | --- |
| Recall@10 | 0.0426 |
| NDCG@10 | 0.0199 |
| Coverage | 0.0125 |
| Long-tail coverage | 0.0000 |
| ILD | 0.7473 |

这是**下限基线**：长尾覆盖率为 0 正是热门推荐的马太效应，后面每一层（ItemCF → BPR → 双塔 → 精排 → 重排）都应该把它推上去。
λ 扫描（valid 子集）也印证了权衡：`λ=0.5` 时 ILD 0.787 / Recall 0.030，`λ=1.0` 时 ILD 0.693 / Recall 0.040。
复现：`scripts/evaluate.py`，结果写入 `artifacts/metrics/`（不入库）。

## Roadmap

| 阶段 | 内容 | 状态 |
| --- | --- | --- |
| 0 | venv、配置、目录、统一协议、脚本 | 已完成 |
| 1 | 下载、时序切分、防穿越断言、负采样 | 已完成 |
| 2 | 热门基线（含时间衰减） | 已完成 |
| 2 | ItemCF：稀疏共现 + 热度惩罚 + top-N 邻居 | 接口占位 |
| 3 | BPR-MF（pairwise loss、负采样、正则）+ ALS/SVD 对照 | 接口占位 |
| 4 | 双塔（in-batch negatives + false-negative mask + 温度系数）、FAISS 索引 | 接口占位 |
| 5 | DIN（行为序列 attention）/ DeepFM 对照，上游分数作为特征 | 接口占位 |
| 6 | MMR 重排 + λ 扫描的相关性/多样性权衡曲线 | 已完成 |
| 7 | Recall@10、NDCG@10、覆盖率、长尾覆盖率 + 消融对比 | 已完成 |
| 8 | FastAPI `/recommend/{user_id}` + 网页 + 冷启动兜底 + 分步耗时 | 已完成 |

每完成一层，在对应类里把 `implemented` 改为 `True` 并实现 `recall` / `predict` 即可自动接入
`RECALL_REGISTRY` / `RANK_REGISTRY`，评估与 API 无需修改。

## 面试主线

每一层要能说清「解决了上一层什么问题、代价是什么」：热门无个性化且有马太效应 → ItemCF 引入协同
但稀疏/冷启动差 → BPR 用隐向量泛化并直接优化排序 → 双塔可加特征且能用 ANN 检索百万候选，
但用户与物品直到最后才交互 → 精排做交叉特征与兴趣建模，只能打几百个候选 → 重排修多样性与长尾 →
评估暴露离线指标与线上目标的错位 → 服务暴露延迟、缓存、降级、embedding 版本一致性。
高频追问：为什么必须时间切分、负采样怎么做、曝光偏差与样本选择偏差（SSB）、精确/近似检索、
冷启动、A/B 与线上指标。

数据源：MovieLens（GroupLens），`ml-latest-small` / `ml-1m`，仅用于学习与研究。
