import psycopg2
import random
import csv
import json
import numpy as np

# 数据库连接配置（DSN 字符串）
dsn = "dbname=TPCH user=postgres password=123456 host=localhost port=5432"
conn = psycopg2.connect(dsn)

# 创建游标
cur = conn.cursor()

# 定义 TPC-H 查询模板
query_templates = [
    "SELECT * FROM orders WHERE o_orderdate > '%s';",  # 查询订单表按日期
    "SELECT * FROM lineitem WHERE l_quantity > %s;",  # 查询lineitem表按数量
    "SELECT * FROM customer WHERE c_mktsegment = '%s';",  # 查询客户表按市场段
    "SELECT * FROM part WHERE p_brand = '%s';",  # 查询part表按品牌
    "SELECT * FROM supplier WHERE s_nationkey = %s;"  # 查询supplier表按国家key
]

# 随机生成查询条件
def generate_random_query():
    template = random.choice(query_templates)
    
    # 根据模板类型生成随机条件
    if "o_orderdate" in template:
        random_date = f"{random.randint(1992, 1998)}-{'%02d' % random.randint(1, 12)}-{'%02d' % random.randint(1, 28)}"
        return template % random_date
    elif "l_quantity" in template:
        random_value = random.randint(1, 100)
        return template % random_value
    elif "c_mktsegment" in template:
        segments = ['BUILDING', 'AUTOMOBILE', 'FURNITURE', 'MACHINERY', 'FOOD']
        return template % random.choice(segments)
    elif "p_brand" in template:
        brands = ['Brand#12', 'Brand#23', 'Brand#34']
        return template % random.choice(brands)
    elif "s_nationkey" in template:
        nation_key = random.randint(1, 25)
        return template % nation_key

# 执行查询并生成 JSON 格式的查询计划，保存到 CSV 文件中
def save_query_plan_to_csv(query, query_id, writer):
    try:
        # 执行 EXPLAIN 命令以获得查询计划
        cur.execute(f"EXPLAIN (FORMAT JSON) {query}")
        plan = cur.fetchone()[0]  # 返回的是一个 JSON 对象
        
        # 如果查询计划是空的，跳过当前查询
        if plan is None:
            print(f"Query plan {query_id} is empty, skipping.")
            return
        
        # 将查询计划转换为 JSON 字符串
        plan_json = json.dumps(plan)
        cardinality = random.randint(1, 10000)  # 随机生成 cardinality

        # 根据查询内容确定表名
        if "orders" in query:
            table_name = "orders,lineitem"
        elif "lineitem" in query:
            table_name = "lineitem"
        elif "customer" in query:
            table_name = "customer"
        elif "part" in query:
            table_name = "part"
        elif "supplier" in query:
            table_name = "supplier"
        else:
            table_name = "unknown"
        
        # 写入 CSV 文件
        writer.writerow([table_name, query_id, plan_json, cardinality])
        print(f"Query plan {query_id} saved.")
    except Exception as e:
        print(f"Error executing query {query_id}: {e}")

# 生成多个查询计划并保存到 CSV 文件，同时生成相应的位图文件
def generate_plans(num_plans, csv_filename, bitmaps_filename, num_materialized_samples=1000):
    with open(csv_filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)  # 使用默认的逗号分隔符
        bitmaps = []
        num_bytes_per_bitmap = int((num_materialized_samples + 7) >> 3)  # 计算每个位图的字节数

        for i in range(num_plans):
            query = generate_random_query()  # 生成随机查询
            save_query_plan_to_csv(query, i, writer)  # 保存查询计划到 CSV 文件

            # 设置每个查询的位图数量，这里假设为2
            num_bitmaps_curr_query = 2
            bitmaps.append(num_bitmaps_curr_query.to_bytes(4, byteorder='little'))  # 添加位图信息
            
            for _ in range(num_bitmaps_curr_query):
                # 生成随机位图数据
                bitmap = np.random.randint(0, 2, size=(num_bytes_per_bitmap * 8,), dtype=np.uint8)
                bitmap_bytes = np.packbits(bitmap)[:num_bytes_per_bitmap]  # 将位图数据打包成字节
                bitmaps.append(bitmap_bytes)  # 保存位图数据
        
        print(f"Total queries saved: {num_plans}")

    # 写入位图数据到 .bitmaps 文件
    with open(bitmaps_filename, 'wb') as bitmaps_file:
        for bitmap in bitmaps:
            bitmaps_file.write(bitmap)
    print(f"Bitmap file saved as {bitmaps_filename}")

# 生成 100 个查询计划并保存到 "train.csv" 文件，同时生成相应的位图文件
generate_plans(100, './query_plans/train.csv', './query_plans/train.bitmaps')

# 关闭游标和连接
cur.close()
conn.close()
