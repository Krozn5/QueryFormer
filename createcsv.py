import psycopg2
import random
import csv
import json
import os
from datetime import datetime

# 数据库连接配置（DSN 字符串）
dsn = "dbname=TPCH user=postgres password=123456 host=localhost port=5432"
conn = psycopg2.connect(dsn)

# 创建游标
cur = conn.cursor()

# 定义 TPC-H 查询模板
query_templates = [
    "SELECT * FROM orders WHERE o_orderdate > %s;",
    "SELECT * FROM lineitem WHERE l_quantity > %s;",
    # 可以添加更多查询模板
]

# 随机生成查询条件
def generate_random_query():
    template = random.choice(query_templates)
    if "o_orderdate" in template:
        random_date = datetime(
            year=random.randint(1992, 1998),
            month=random.randint(1, 12),
            day=random.randint(1, 28)
        ).strftime('%Y-%m-%d')
        return template, [random_date]
    elif "l_quantity" in template:
        random_value = random.randint(1, 100)
        return template, [random_value]

# 执行查询并生成 JSON 格式的查询计划，保存到 CSV
def save_query_plan_to_csv(query, params, query_id, writer):
    try:
        # 使用 EXPLAIN (ANALYZE, FORMAT JSON) 获取查询计划
        cur.execute(f"EXPLAIN (ANALYZE, FORMAT JSON) {query}", params)
        plan = cur.fetchone()[0][0]  # 获取完整的计划，包括 'Plan'、'Planning Time'、'Execution Time' 等

        # 转换为字符串格式
        plan_json_str = json.dumps(plan, ensure_ascii=False)

        # 写入 CSV，包含两列：id 和 json
        writer.writerow([query_id, plan_json_str])
        print(f"Query plan {query_id} saved.")
    except Exception as e:
        print(f"Error executing query {query_id}: {e}")
        with open("error_log.txt", mode='a', encoding='utf-8') as log_file:
            log_file.write(f"Query ID {query_id}: {e}\n")

# 生成多个查询计划并保存到 CSV 文件
def generate_plans(num_plans, csv_filename):
    # 确保目标目录存在
    os.makedirs(os.path.dirname(csv_filename), exist_ok=True)
    
    with open(csv_filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['id', 'json'])  # 写入表头
        for i in range(num_plans):
            query, params = generate_random_query()
            save_query_plan_to_csv(query, params, i, writer)

# 主程序入口
if __name__ == "__main__":
    # 生成 100 个查询计划并保存到指定 CSV 文件
    csv_path = 'E:/QueryFormer-main/query_plans/train14.csv'  # 使用绝对路径
    generate_plans(100, csv_path)

    # 关闭游标和连接
    cur.close()
    conn.close()
