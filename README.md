# 换签期限前的货架余量估计与盘点

当前资料覆盖包装版别收发存、盘点与撤架时段。记录均为合成数据，使用 UTF-8 编码及带列名的 CSV。

## 运行

使用 Python 3.11。分析依赖在 `requirements.txt` 中锁定，本地环境可运行 `python3 -m pip install -r requirements.txt` 安装。在项目根目录查看文件列和记录数，并核对交接格式。

```sh
python3 inspect_inputs.py data
python3 -m unittest discover -s tests -v
```

## 记录约定

收银按商品码，不能判断旧新版。transfer 的取走和送达按物理交接时间，数量以件计，其候选旧版上下界随记录给出。期限生效时库存中的旧版退出正规货架，新版可照常售。库位待验退货不含在可售盘点。historical 中 observed 是收银实销，available_until 早于闭店时不能把后半日当零需求。

- `opening.csv` 期初库存版别范围，数量为同一时刻的总件数。
- `movements.csv` 有物理效力的收货、收银、退货和撤签记录。
- `transfer.csv` 调拨实物在途，known 为空表示收店交接未确认。
- `counts.csv` 实盘记录，包含合成盘点的旧新版分量。
- `slots.csv` 盘点与撤架资源，周转车按给定时段最多容纳 max_units 件。
- `historical.csv` 已结束的多期本地换签历史。
- `deadlines.csv` 门店撤签生效、盘点确认期限及可接受残余风险。
