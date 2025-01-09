# %% 导入必要的库
import numpy as np
import os
import torch
import torch.nn as nn
import pandas as pd
import json
from model.util import Normalizer
from model.model import QueryFormer
from model.database_util import Encoding, formatFilter, formatJoin, TreeNode, filterDict2Hist, get_hist_file, collator, get_job_table_sample, Batch
from model.dataset import PlanTreeDataset
from model.trainer import train
import torch.serialization

# %% 定义数据路径和其他参数
data_path = './data/imdb/test/'  # 数据文件所在的路径
model_path = './results/full/cost/955.pt'  # 训练好的模型路径

# %% 确保结果保存目录存在
class Args:
    bs = 128  # 批量大小
    lr = 0.001  # 学习率
    epochs = 100  # 训练轮数
    clip_size = 50  # 梯度裁剪大小
    embed_size = 64  # 嵌入层大小（与之前代码一致）
    pred_hid = 128  # 隐藏层大小（与之前代码一致）
    ffn_dim = 128  # FFN 层维度（与之前代码一致）
    head_size = 12  # 注意力头的数量（与之前代码一致）
    n_layers = 8  # Transformer 的层数
    dropout = 0.1  # Dropout 比率
    sch_decay = 0.6  # 学习率衰减系数
    device = 'cuda:0'  # 使用 GPU
    newpath = './results/full/cost/'  # 模型保存路径
    to_predict = 'cost'  # 要预测的值是“cost”

args = Args()

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
torch.serialization.add_safe_globals([Encoding])
encoding_ckpt = torch.load('checkpoints/encoding.pt', map_location='cpu')
encoding = encoding_ckpt['encoding']
print("提示：编码信息加载完成")

# %% 初始化模型
print("提示：正在加载训练好的模型...")
checkpoint = torch.load(model_path, map_location='cpu')
args = checkpoint['args']  # 使用保存的训练参数
model = QueryFormer(emb_size=args.embed_size, ffn_dim=args.ffn_dim, head_size=args.head_size,
                    dropout=args.dropout, n_layers=args.n_layers,
                    use_sample=True, use_hist=True, pred_hid=args.pred_hid)

# 手动删除不匹配的权重
state_dict = checkpoint['model']
model_state_dict = model.state_dict()
incompatible_keys = [key for key in state_dict if key not in model_state_dict or state_dict[key].shape != model_state_dict[key].shape]
for key in incompatible_keys:
    del state_dict[key]

model.load_state_dict(state_dict, strict=False)
model = model.to(args.device)
model.eval()
print("提示：模型加载完成")

# %% 加载下游任务的数据（例如，推理集）
print("提示：正在加载推理数据（train2.csv）...")
infer_df = pd.read_csv(data_path + 'train2.csv')

# 不对 infer_df['json'] 进行任何处理，保持与训练时一致
print(f"提示：推理数据加载完成，共包含 {len(infer_df)} 条记录")

# %% 加载 job 表样本
print("提示：正在加载 job 表样本...")
table_sample = get_job_table_sample(data_path + 'train')
print("提示：job 表样本加载完成")

# %% 创建推理数据集
print("提示：正在创建推理数据集...")
infer_ds = PlanTreeDataset(infer_df, None, encoding, hist_file, card_norm, cost_norm, args.to_predict, table_sample)
print("提示：推理数据集创建完成")

# %% 定义推理函数
def inference(model, dataset, device):
    """使用训练好的模型进行推理"""
    results = []
    with torch.no_grad():
        for idx in range(len(dataset)):
            batch_dict, _ = dataset[idx]
            # 检查并调整张量的维度
            attn_bias = batch_dict['attn_bias']
            if attn_bias.dim() == 2:
                attn_bias = attn_bias.unsqueeze(0).to(device)  # 添加批量维度
            else:
                attn_bias = attn_bias.to(device)

            rel_pos = batch_dict['rel_pos']
            if rel_pos.dim() == 2:
                rel_pos = rel_pos.unsqueeze(0).to(device)
            else:
                rel_pos = rel_pos.to(device)

            heights = batch_dict['heights']
            if heights.dim() == 1:
                heights = heights.unsqueeze(0).to(device)
            else:
                heights = heights.to(device)

            x = batch_dict['x']
            if x.dim() == 2:
                x = x.unsqueeze(0).to(device)
            else:
                x = x.to(device)

            batch = Batch(
                attn_bias=attn_bias,
                rel_pos=rel_pos,
                heights=heights,
                x=x
            )
            cost_pred, _ = model(batch)
            cost_pred = cost_pred.item()
            unnormalized_cost = cost_norm.unnormalize_labels([cost_pred])[0]
            results.append(unnormalized_cost)
    return results

# %% 使用训练好的模型进行推理
print("提示：正在使用模型进行推理...")
results = inference(model, infer_ds, args.device)

# %% 保存推理结果
output_path = os.path.join(args.newpath, 'results.csv')
infer_df['Predicted_Cost'] = results
infer_df.to_csv(output_path, index=False)
print(f"提示：推理结果已保存到 {output_path}")
