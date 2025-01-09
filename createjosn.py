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
    "SELECT * FROM orders WHERE o_orderdate > '%s';",
    "SELECT * FROM lineitem WHERE l_quantity > %s;",
    # 可以添加更多查询模板
]

# 随机生成查询条件
def generate_random_query():
    template = random.choice(query_templates)
    if "o_orderdate" in template:
        # 生成随机日期
        random_date = f"{random.randint(1992, 1998)}-{'%02d' % random.randint(1, 12)}-{'%02d' % random.randint(1, 28)}"
        return template % random_date
    elif "l_quantity" in template:
        # 生成随机数量
        random_value = random.randint(1, 100)
        return template % random_value

# 执行查询并生成 JSON 格式的查询计划，保存到 CSV 文件中
def save_query_plan_to_csv(query, query_id, writer):
    try:
        cur.execute(f"EXPLAIN (FORMAT JSON) {query}")
        plan = cur.fetchone()[0]  # 返回的是一个 JSON 对象
        plan_json = json.dumps(plan)  # 转换为 JSON 字符串
        cardinality = random.randint(1, 10000)  # 随机生成的 cardinality (可以根据需求调整)
        table_name = "orders" if "orders" in query else "lineitem"  # 根据查询模板确定表名
        writer.writerow([table_name, query_id, plan_json, cardinality])  # 将计划和其他信息写入 CSV
        print(f"Query plan {query_id} saved.")
    except Exception as e:
        print(f"Error executing query {query_id}: {e}")

# 生成多个查询计划并保存到 CSV 文件，并生成相应的位图文件
def generate_plans(num_plans, csv_filename, bitmaps_filename):
    # 打开 CSV 文件
    with open(csv_filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['table', 'id', 'json', 'cardinality'])  # 写入表头
        bitmaps = []
        
        # 生成查询计划
        for i in range(num_plans):
            query = generate_random_query()
            save_query_plan_to_csv(query, i, writer)
            
            # 生成位图数据 (模拟随机位图，可以根据需求调整)
            bitmap_size = 128  # 假设位图大小为128
            bitmap = np.random.randint(0, 2, size=(bitmap_size,), dtype=np.uint8)
            bitmaps.append(bitmap)
        
    # 保存位图数据到 .bitmaps 文件
    with open(bitmaps_filename, 'wb') as bitmaps_file:
        for bitmap in bitmaps:
            bitmaps_file.write(bitmap.tobytes())
    print(f"Bitmap file saved as {bitmaps_filename}")

# 生成 100 个查询计划并保存到 "query_plans.csv" 文件，同时生成相应的位图文件
generate_plans(100, './query_plans/query_plans.csv', './query_plans/query_plans.bitmaps')

# 关闭游标和连接
cur.close()
conn.close()
