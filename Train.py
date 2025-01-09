import psycopg2
import torch
import pandas as pd
from model.model import QueryFormer
from model.database_util import Encoding
from model.dataset import PlanTreeDataset
from model.trainer import train
import numpy as np
import os
import torch.nn as nn
from model.util import Normalizer
from model.database_util import get_hist_file, get_job_table_sample, collator

# 数据库连接和数据加载
def load_data_from_db(query):
    """ 从数据库加载数据 """
    conn = psycopg2.connect(
        dbname="TPCH", 
        user="postgres", 
        password="123456", 
        host="localhost", 
        port="5432"
    )
    return pd.read_sql(query, conn)

# 从数据库中生成查询计划
def generate_plan_from_db(cursor, query):
    """
    使用 EXPLAIN ANALYZE 生成查询计划并返回 JSON 格式的计划。
    """
    cursor.execute(f"EXPLAIN (ANALYZE, FORMAT JSON) {query}")
    result = cursor.fetchall()
    return result[0][0]  # 假设 EXPLAIN 返回的是一个 JSON 格式的查询计划

def generate_histograms_from_db(cursor, table, column, bins=50):
    # 从数据库获取数据
    query = f"SELECT {column} FROM {table}"
    cursor.execute(query)
    values = cursor.fetchall()

    # 处理从数据库中获取的值，确保它们是有效的数字
    values = [v[0] for v in values if isinstance(v[0], (int, float)) and not np.isnan(v[0])]
    
    if len(values) == 0:
        print(f"警告：表 {table} 列 {column} 中没有有效数据")
        return [], []

    # 确保 'bins' 是一个数值列表
    if isinstance(bins, int):
        min_val, max_val = min(values), max(values)
        bin_edges = np.linspace(min_val, max_val, bins + 1)
    else:
        bin_edges = bins

    # 计算直方图
    hist, bin_edges = np.histogram(values, bins=bin_edges, density=True)

    # 返回直方图和 bin_edges
    return hist, bin_edges


# 从数据库生成样本
def generate_table_samples(cursor, table_name, sample_size=1000):
    """
    从数据库中生成采样。
    """
    query = f"SELECT * FROM {table_name} LIMIT {sample_size}"
    cursor.execute(query)
    rows = cursor.fetchall()
    return np.array(rows)

# 定义训练的参数
class Args:
    bs = 128  # 批量大小
    lr = 0.001  # 学习率
    epochs = 100  # 训练轮数
    clip_size = 50  # 梯度裁剪大小
    embed_size = 64  # 嵌入层大小
    pred_hid = 128  # 隐藏层大小
    ffn_dim = 128  # FFN 层维度
    head_size = 12  # 注意力头的数量
    n_layers = 8  # Transformer 的层数
    dropout = 0.1  # Dropout 比率
    sch_decay = 0.6  # 学习率衰减系数
    device = 'cuda:0'  # 使用 GPU
    newpath = './results/full/cost/'  # 模型保存路径
    to_predict = 'cost'  # 要预测的值是“cost”（成本）
args = Args()

# 确保结果保存目录存在
if not os.path.exists(args.newpath):
    os.makedirs(args.newpath)
    print(f"提示：创建了保存模型的目录 {args.newpath}")
else:
    print(f"提示：保存模型的目录 {args.newpath} 已存在")

# 连接数据库并加载数据
conn = psycopg2.connect(dbname="TPCH", user="postgres", password="123456", host="localhost", port="5432")
cursor = conn.cursor()

# 生成训练集和验证集查询计划
train_query = """
SELECT o.o_orderkey, o.o_orderdate, l.l_partkey, l.l_quantity
FROM orders o
JOIN lineitem l ON o.o_orderkey = l.l_orderkey
WHERE o.o_orderdate > '1995-01-01';
"""
train_plan = generate_plan_from_db(cursor, train_query)

val_query = """
SELECT o.o_orderkey, o.o_orderdate, l.l_partkey, l.l_quantity
FROM orders o
JOIN lineitem l ON o.o_orderkey = l.l_orderkey
WHERE o.o_orderdate > '1994-01-01' AND o.o_orderdate <= '1995-01-01';
"""
val_plan = generate_plan_from_db(cursor, val_query)

# 生成直方图数据
hist_file = []
for table in ['lineitem', 'orders']:
    for column in ['l_quantity', 'o_totalprice']:
        hist_file.append(generate_histograms_from_db(cursor, table, column))

hist_file = pd.DataFrame(hist_file)

# 生成表样本数据
table_sample = generate_table_samples(cursor, 'lineitem', 1000)

# 加载编码信息
encoding_ckpt = torch.load('query_plans/encoding1.pt')
encoding = encoding_ckpt['encoding']

# 初始化模型
print("提示：正在初始化模型...")
model = QueryFormer(
    emb_size=args.embed_size, ffn_dim=args.ffn_dim, head_size=args.head_size,
    dropout=args.dropout, n_layers=args.n_layers,
    use_sample=True, use_hist=True, pred_hid=args.pred_hid
)

# 将模型移动到设备
model = model.to(args.device)
print(f"提示：模型已初始化并移动到设备 {args.device}")

# 创建训练集和验证集
print("提示：正在创建训练集和验证集的数据集...")
train_df = pd.DataFrame({'json': [train_plan]})  # 这里使用查询计划生成 DataFrame
val_df = pd.DataFrame({'json': [val_plan]})

# 这里用静态示例代替数据集加载，实际应根据你的数据构建
train_ds = PlanTreeDataset(train_df, None, encoding, hist_file, cost_norm, card_norm, args.to_predict, table_sample)
val_ds = PlanTreeDataset(val_df, None, encoding, hist_file, cost_norm, card_norm, args.to_predict, table_sample)
print("提示：数据集创建完成")

# 定义损失函数
crit = nn.MSELoss()

# 开始训练模型
print(f"提示：开始训练模型，共 {args.epochs} 轮...")
model, best_path = train(model, train_ds, val_ds, crit, cost_norm, args)
print(f"提示：训练完成，模型已保存到 {best_path}")

# 保存最终模型
model_save_path = os.path.join(args.newpath, '955.pt')
torch.save({'model': model.state_dict(), 'args': args}, model_save_path)
print(f"提示：最终模型已保存到 {model_save_path}")

# 关闭数据库连接
cursor.close()
conn.close()
