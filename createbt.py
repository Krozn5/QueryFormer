import psycopg2
import torch
from model.database_util import Encoding

# 数据库连接
conm = psycopg2.connect(database="TPCH", user="postgres", host="localhost", password="123456", port="5432")
conm.set_session(autocommit=True)
cur = conm.cursor()

# TPC-H 表结构和别名
tpch_schema = {
    'customer': [
        'c.c_custkey', 'c.c_name', 'c.c_address', 'c.c_nationkey', 'c.c_phone',
        'c.c_acctbal', 'c.c_mktsegment', 'c.c_comment'
    ],
    'lineitem': [
        'l.l_orderkey', 'l.l_partkey', 'l.l_suppkey', 'l.l_linenumber', 'l.l_quantity',
        'l.l_extendedprice', 'l.l_discount', 'l.l_tax', 'l.l_returnflag', 'l.l_linestatus',
        'l.l_shipdate', 'l.l_commitdate', 'l.l_receiptdate', 'l.l_shipinstruct',
        'l.l_shipmode', 'l.l_comment'
    ],
    'nation': [
        'n.n_nationkey', 'n.n_name', 'n.n_regionkey', 'n.n_comment'
    ],
    'orders': [
        'o.o_orderkey', 'o.o_custkey', 'o.o_orderstatus', 'o.o_totalprice',
        'o.o_orderdate', 'o.o_orderpriority', 'o.o_clerk', 'o.o_shippriority', 'o.o_comment'
    ],
    'part': [
        'p.p_partkey', 'p.p_name', 'p.p_mfgr', 'p.p_brand', 'p.p_type',
        'p.p_size', 'p.p_container', 'p.p_retailprice', 'p.p_comment'
    ],
    'partsupp': [
        'ps.ps_partkey', 'ps.ps_suppkey', 'ps.ps_availqty', 'ps.ps_supplycost', 'ps.ps_comment'
    ],
    'region': [
        'r.r_regionkey', 'r.r_name', 'r.r_comment'
    ],
    'supplier': [
        's.s_suppkey', 's.s_name', 's.s_address', 's.s_nationkey', 's.s_phone',
        's.s_acctbal', 's.s_comment'
    ]
}

# 为每个列设置最小值和最大值
column_min_max_vals = {}

for table, columns in tpch_schema.items():
    table_alias = table[0]  # 例如 'customer' -> 'c'
    for column in columns:
        column_name = column.split('.')[1]  # 例如 'c.c_custkey' -> 'c_custkey'
        query = f"SELECT MIN({column_name}), MAX({column_name}) FROM {table}"
        cur.execute(query)
        min_val, max_val = cur.fetchone()
        if min_val is not None and max_val is not None:
            column_min_max_vals[column] = (min_val, max_val)
        else:
            # 如果查询结果为空，设定默认值
            column_min_max_vals[column] = (0, 1)

# 建立列和索引的对应关系，使用表别名
col2idx = {}
idx = 0

for table, columns in tpch_schema.items():
    for column in columns:
        col2idx[column] = idx  # 使用 'l.l_quantity' 格式的键
        idx += 1

# 创建 Encoding 对象并保存
encoding = Encoding(
    column_min_max_vals=column_min_max_vals,
    col2idx=col2idx,
    op2idx={'>': 0, '=': 1, '<': 2, 'NA': 3}
)

# 保存到 encoding1.pt 文件
save_path = 'E:/QueryFormer-main/query_plans/encoding1.pt'
torch.save({'encoding': encoding}, save_path)
print(f'Encoding 文件已保存到 {save_path}')

# 验证保存的文件是否可以被正确加载
encoding_ckpt = torch.load(save_path)
encoding = encoding_ckpt['encoding']
print("编码信息加载完成")
