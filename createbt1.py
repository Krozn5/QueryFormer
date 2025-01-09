import numpy as np
import pandas as pd
import csv
import torch
import psycopg2
import datetime  # 添加 datetime 模块的导入
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

# 定义获取列数据类型的函数
def get_column_data_type(table_name, column_name):
    query = f"""
    SELECT data_type
    FROM information_schema.columns
    WHERE table_name='{table_name}' AND column_name='{column_name}'
    """
    cur.execute(query)
    result = cur.fetchone()
    if result:
        return result[0]
    else:
        return None

# 为每个列设置最小值和最大值
column_min_max_vals = {}

for table, columns in tpch_schema.items():
    for column in columns:
        # 提取表别名和列名，例如 'l.l_quantity' -> 'l' 和 'l_quantity'
        column_alias, column_name = column.split('.')
        # 去除列名前的别名，例如 'l_quantity' -> 'l_quantity'
        column_name = column_name  # 已经是列名，无需处理
        # 使用完整表名构建键，例如 'lineitem.l_quantity'
        full_column_name = f"{table}.{column_name}"

        # 获取列的数据类型
        data_type = get_column_data_type(table, column_name)

        if data_type is None:
            print(f"警告：无法找到表 {table} 中列 {column_name} 的数据类型，跳过该列。")
            continue

        # 根据数据类型处理
        if data_type in ('integer', 'numeric', 'real', 'double precision', 'smallint', 'bigint', 'decimal'):
            # 数值类型
            query = f"SELECT MIN({column_name}), MAX({column_name}) FROM {table}"
            cur.execute(query)
            min_val, max_val = cur.fetchone()
            if min_val is not None and max_val is not None:
                column_min_max_vals[full_column_name] = (float(min_val), float(max_val))
            else:
                # 如果查询结果为空，设定默认值
                column_min_max_vals[full_column_name] = (0.0, 1.0)
        elif data_type in ('date', 'timestamp', 'timestamp without time zone', 'timestamp with time zone'):
            # 日期类型
            query = f"SELECT MIN({column_name}), MAX({column_name}) FROM {table}"
            cur.execute(query)
            min_val, max_val = cur.fetchone()
            if min_val is not None and max_val is not None:
                column_min_max_vals[full_column_name] = (min_val, max_val)
            else:
                # 如果查询结果为空，设定默认日期范围
                column_min_max_vals[full_column_name] = (datetime.date(1970, 1, 1), datetime.date(1970, 1, 2))
        else:
            # 非数值类型（字符串、文本），跳过
            print(f"跳过非数值列 {column_name}（类型：{data_type}）")
            continue

# 建立列和索引的对应关系，使用完整的表名
col2idx = {}
idx = 0

for table, columns in tpch_schema.items():
    for column in columns:
        # 提取表别名和列名，例如 'l.l_quantity' -> 'l' 和 'l_quantity'
        column_alias, column_name = column.split('.')
        # 使用完整表名构建键，例如 'lineitem.l_quantity'
        full_column_name = f"{table}.{column_name}"
        # 检查该列是否在 column_min_max_vals 中（即是否为数值或日期类型）
        if full_column_name in column_min_max_vals:
            col2idx[full_column_name] = idx  # 使用 'lineitem.l_quantity' 作为键
            idx += 1
        else:
            # 跳过非数值列
            continue

# 添加 'NA' 列到 col2idx
col2idx['NA'] = idx

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

# 检查 'lineitem.l_quantity' 是否在 col2idx 中
if 'lineitem.l_quantity' in encoding.col2idx:
    print("成功包含 'lineitem.l_quantity' 列。")
else:
    print("未找到 'lineitem.l_quantity' 列，请检查编码生成过程。")
