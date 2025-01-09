import pandas as pd
import numpy as np
import psycopg2
import json

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
t2alias = {
    'customer': 'c', 'lineitem': 'l', 'nation': 'n', 'orders': 'o',
    'part': 'p', 'partsupp': 'ps', 'region': 'r', 'supplier': 's'
}

# 数据库连接
conm = psycopg2.connect(database="TPCH", user="postgres", host="localhost", password="123456", port="5432")
conm.set_session(autocommit=True)
cur = conm.cursor()


def to_vals(data_list):
    """
    将数据库数据转换为数值或时间戳数组。
    """
    res = []
    for dat in data_list:
        val = dat[0]
        try:
            res.append(float(val))
        except (ValueError, TypeError):
            try:
                res.append(val.timestamp() if hasattr(val, 'timestamp') else 0)
            except AttributeError:
                res.append(0)
    return np.array(res, dtype=np.float64)


# 初始化直方图文件
hist_file = pd.DataFrame(columns=['table', 'column', 'table_column', 'freq', 'bins'])

# 生成直方图数据
for table, columns in tpch_schema.items():
    for column in columns:
        try:
            cmd = f'SELECT {column.split(".")[1]} FROM {table} AS {t2alias[table]}'
            cur.execute(cmd)
            col = cur.fetchall()
            col_array = to_vals(col)

            # 检查数据是否为空
            if col_array.size == 0:
                continue

            # 计算频率分布
            freq, _ = np.histogram(col_array, bins=50)
            freq_hex = freq.tobytes().hex()  # 转换为十六进制字符串

            # 计算分箱边界
            hists = np.nanpercentile(col_array, range(0, 101, 2))

            # 构建记录
            res_dict = {
                'table': table,
                'column': column.split(".")[1],
                'table_column': f'{t2alias[table]}.{column.split(".")[1]}',
                'freq': freq_hex,
                'bins': ','.join(map(str, hists))  # 将分箱边界保存为逗号分隔的字符串
            }
            hist_file = pd.concat([hist_file, pd.DataFrame([res_dict])], ignore_index=True)
        except Exception as e:
            print(f"处理 {table}.{column} 时出错：{e}")

# 保存文件
output_path = r"E:\QueryFormer-main\query_plans\histogram_string.csv"
hist_file.to_csv(output_path, index=False)
print(f"提示：直方图文件已保存至 {output_path}")
