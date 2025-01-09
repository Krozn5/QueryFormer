# %% 导入必要的库
import numpy as np
import os
import torch
import torch.nn as nn
import pandas as pd
from model.util import Normalizer
from model.database_util import get_hist_file, get_job_table_sample, collator
from model.model import QueryFormer
from model.database_util import Encoding
from model.dataset import PlanTreeDataset
from model.trainer import train

# %% 定义数据路径和其他参数
data_path = './query_plans/'  # 数据文件所在的路径

# %% 定义训练的参数
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

# %% 确保结果保存目录存在
if not os.path.exists(args.newpath):
    os.makedirs(args.newpath)
    print(f"提示：创建了保存模型的目录 {args.newpath}")
else:
    print(f"提示：保存模型的目录 {args.newpath} 已存在")

# %% 加载直方图文件
print("提示：正在加载直方图文件...")
hist_file = get_hist_file(data_path + 'histogram_string.csv')
print("提示：直方图文件加载完成")

# 初始化归一化器，用于成本和基数的归一化和反归一化
cost_norm = Normalizer(-3.61192, 12.290855)
card_norm = Normalizer(1, 100)

# %% 加载编码信息
print("提示：正在加载编码信息...")
encoding_ckpt = torch.load('query_plans/encoding1.pt')
encoding = encoding_ckpt['encoding']
print("提示：编码信息加载完成")

# %% 初始化模型
print("提示：正在初始化模型...")
model = QueryFormer(emb_size=args.embed_size, ffn_dim=args.ffn_dim, head_size=args.head_size,
                    dropout=args.dropout, n_layers=args.n_layers,
                    use_sample=True, use_hist=True, pred_hid=args.pred_hid)

# 将模型移动到指定设备（GPU/CPU）
_ = model.to(args.device)
print(f"提示：模型已初始化并移动到设备 {args.device}")

# %% 加载训练数据
print("提示：正在加载训练数据（train1.csv 和 train2.csv）...")
train_dfs = []
for i in range(11, 13):  # 加载 train1.csv 和 train2.csv 作为训练集
    file = data_path + f'train{i}.csv'
    df = pd.read_csv(file)
    train_dfs.append(df)

# 合并所有训练数据
full_train_df = pd.concat(train_dfs)
print(f"提示：训练数据加载完成，共包含 {len(full_train_df)} 条记录")

# %% 加载验证数据
print("提示：正在加载验证数据（train3.csv 和 train4.csv）...")
val_dfs = []
for i in range(13, 15):  # 加载 train3.csv 和 train4.csv 作为验证集
    file = data_path + f'train{i}.csv'
    df = pd.read_csv(file)
    val_dfs.append(df)

# 合并所有验证数据
val_df = pd.concat(val_dfs)
print(f"提示：验证数据加载完成，共包含 {len(val_df)} 条记录")

# %% 加载 job 表样本
print("提示：正在加载 job 表样本...")
try:
    table_sample = get_job_table_sample(data_path + 'train')
    if not table_sample:  # 如果没有有效数据，设置一个空的默认值
        print("警告：job 表样本为空，将使用默认空样本")
        table_sample = []  # 设置默认值
    else:
        print("提示：job 表样本加载完成，样本数量：", len(table_sample))
except Exception as e:
    print("错误：加载 job 表样本时出现问题：", e)



# %% 创建训练集和验证集的 Dataset
print("提示：正在创建训练集和验证集的数据集...")
train_ds = PlanTreeDataset(full_train_df, None, encoding, hist_file, card_norm, cost_norm, args.to_predict, table_sample)
val_ds = PlanTreeDataset(val_df, None, encoding, hist_file, card_norm, cost_norm, args.to_predict, table_sample)
print("提示：数据集创建完成")

# %% 定义损失函数
crit = nn.MSELoss()

# %% 开始训练模型
print(f"提示：开始训练模型，共 {args.epochs} 轮...")
model, best_path = train(model, train_ds, val_ds, crit, cost_norm, args)
print(f"提示：训练完成，模型已保存到 {best_path}")

# %% 保存最终模型
model_save_path = os.path.join(args.newpath, '955.pt')
torch.save({'model': model.state_dict(), 'args': args}, model_save_path)
print(f"提示：最终模型已保存到 {model_save_path}")
