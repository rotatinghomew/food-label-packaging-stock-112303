# 换签期限前的货架余量估计与盘点

当前资料覆盖包装版别收发存、盘点与撤架时段。记录均为合成数据，使用 UTF-8 编码及带列名的 CSV。

## 运行

使用 Python 3.11。分析依赖在 `requirements.txt` 中锁定，本地环境可运行 `python3 -m pip install -r requirements.txt` 安装。在项目根目录查看文件列和记录数，并核对交接格式。

```sh
python3 inspect_inputs.py data
python3 -m unittest discover -s tests -v
python3 solve.py data --as-of 2026-09-05T08:00 --out outputs
```

`solve.py` 会生成：

- `outputs/analysis.json`：门店存量区间、需求回测、盘点/车辆方案、同批货轨迹。
- `outputs/report.md`：给区域经理看的中文汇总。

也可从 Python 调用：

```python
from analysis import analyze, old_ranges

report = analyze("data", "2026-09-05T08:00")
ranges = old_ranges("data")
```

## 分析模型

- `relabel/ledger.py` 用全局线性可行流同时约束期初版本界、混叠收银、收退货、
  调拨共享旧版比例和实盘旧/新分量；所有时点后均要求旧版、新版非负。
- 待验退货进入检查库位，不进入可售总量，也不能用于解释收银缺口。
- 调拨在 `picked → arrived` 之间只作为在途：原店已扣减，收货店未增加；
  `transfer.known` 只改变交接确认状态。
- `relabel/forecast.py` 按 `observed / (available_until-opens)` 估计有效暴露速度，
  短日不补零，并提供 16 个完整 trial 的滚动/留一期回测。
- 残余范围拆成当前存量宽度和未来需求宽度；风险预留满足
  `P(最坏口径残余 > reserve) <= max_residual_probability`。
- 只有本期 `old=0` 实盘且之后没有可能含旧版的入库，才能证明某店免撤；
  自然卖完预测或 LP 下界为 0 均不能替代实盘。
- 已撤下数量按事件时间永久扣减；晚到的 `known` 销售只进入历史回算，
  不会在当前状态中把已撤下旧签加回。

## 当前资料在 2026-09-05 08:00 的要点

- A：可售 60，n1 已证明旧版 43、新版 17。
- B：可售 92，x2 调出后旧版 19–33；下限是跨所有可行流可确认的旧签，
  宽度 14 来自 x2 旧版比例未分清。
- C：可售 125，旧版 32–91，宽度 59；s3 的存量信息价值最大。
- s2 可厘清 B/x2；若 K1 同一班组且 s2、s3 冲突，按更早期限选 s2，
  C 需另找盘点时段。若门店可并行自查，则 s1、s2、s3 均可安排。
- x2 交接未确认，即使预测 B 可卖完，也不能把 B 的撤签预约按零取消。

## 记录约定

收银按商品码，不能判断旧新版。transfer 的取走和送达按物理交接时间，数量以件计，其候选旧版上下界随记录给出。期限生效时库存中的旧版退出正规货架，新版可照常售。库位待验退货不含在可售盘点。historical 中 observed 是收银实销，available_until 早于闭店时不能把后半日当零需求。

- `opening.csv` 期初库存版别范围，数量为同一时刻的总件数。
- `movements.csv` 有物理效力的收货、收银、退货和撤签记录。
- `transfer.csv` 调拨实物在途，known 为空表示收店交接未确认。
- `counts.csv` 实盘记录，包含合成盘点的旧新版分量。
- `slots.csv` 盘点与撤架资源，周转车按给定时段最多容纳 max_units 件。
- `historical.csv` 已结束的多期本地换签历史。
- `deadlines.csv` 门店撤签生效、盘点确认期限及可接受残余风险。
