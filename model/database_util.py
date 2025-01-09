import numpy as np
import pandas as pd
import csv
import torch
import json
import datetime

# BFS 应该足够
def floyd_warshall_rewrite(adjacency_matrix):
    (nrows, ncols) = adjacency_matrix.shape
    assert nrows == ncols
    M = adjacency_matrix.copy().astype('long')
    for i in range(nrows):
        for j in range(ncols):
            if i == j:
                M[i][j] = 0
            elif M[i][j] == 0:
                M[i][j] = 60

    for k in range(nrows):
        for i in range(nrows):
            for j in range(nrows):
                M[i][j] = min(M[i][j], M[i][k] + M[k][j])
    return M

def get_job_table_sample(workload_file_name, num_materialized_samples=1000):
    tables = []
    samples = []

    # 加载查询
    with open(workload_file_name + ".csv", 'r') as f:
        data_raw = list(list(rec) for rec in csv.reader(f, delimiter='#'))
        for row in data_raw:
            tables.append(row[0].split(','))

            if int(row[3]) < 1:
                print("Queries must have non-zero cardinalities")
                exit(1)

    print("Loaded queries with len ", len(tables))

    # 加载位图
    num_bytes_per_bitmap = int((num_materialized_samples + 7) >> 3)
    with open(workload_file_name + ".bitmaps", 'rb') as f:
        for i in range(len(tables)):
            four_bytes = f.read(4)
            if not four_bytes:
                print("Error while reading 'four_bytes'")
                exit(1)
            num_bitmaps_curr_query = int.from_bytes(four_bytes, byteorder='little')
            bitmaps = np.empty((num_bitmaps_curr_query, num_bytes_per_bitmap * 8), dtype=np.uint8)
            for j in range(num_bitmaps_curr_query):
                # 读取位图
                bitmap_bytes = f.read(num_bytes_per_bitmap)
                if not bitmap_bytes:
                    print("Error while reading 'bitmap_bytes'")
                    exit(1)
                bitmaps[j] = np.unpackbits(np.frombuffer(bitmap_bytes, dtype=np.uint8))
            samples.append(bitmaps)
    print("Loaded bitmaps")
    table_sample = []
    for ts, ss in zip(tables, samples):
        d = {}
        for t, s in zip(ts, ss):
            tf = t.split(' ')[0]  # 去除别名
            d[tf] = s
        table_sample.append(d)

    return table_sample

def get_hist_file(hist_path, bin_number=50):
    hist_file = pd.read_csv(hist_path)

    # 将 'freq' 列从十六进制字符串转换为 numpy 数组
    for i in range(len(hist_file)):
        freq = hist_file.at[i, 'freq']
        try:
            # 尝试从十六进制字符串转换为浮动数值
            freq_np = np.frombuffer(bytes.fromhex(freq), dtype=float)
            hist_file.at[i, 'freq'] = freq_np
        except Exception as e:
            print(f"警告：无法处理列 {i} 的 freq 值 '{freq}': {e}")
            hist_file.at[i, 'freq'] = np.array([])  # 如果转换失败，将该列设为一个空数组

    # 更新 'table_column' 的值
    table_column = []
    for i in range(len(hist_file)):
        table = hist_file.at[i, 'table']
        col = hist_file.at[i, 'column']
        # 使用完整的表名
        combine = f"{table}.{col}"
        table_column.append(combine)
    hist_file['table_column'] = table_column

    # 将 'bins' 列从逗号分隔的字符串转换为浮点数列表
    for rid in range(len(hist_file)):
        bins_str = hist_file.at[rid, 'bins']
        try:
            # 确保所有项都可以转换为浮动数值
            bins_list = [float(i) for i in bins_str.split(',') if len(i) > 0]
            hist_file.at[rid, 'bins'] = bins_list
        except Exception as e:
            print(f"警告：无法处理列 {rid} 的 bins 值 '{bins_str}': {e}")
            hist_file.at[rid, 'bins'] = []  # 如果解析失败，设置为空列表

    # 删除重复的 table_column 值
    hist_file = hist_file.drop_duplicates(subset=['table_column'])

    # 如果需要重新分箱
    if bin_number != 50:
        hist_file = re_bin(hist_file, bin_number)

    return hist_file

def re_bin(hist_file, target_number):
    for i in range(len(hist_file)):
        freq = hist_file['freq'][i]
        if len(freq) > 0:  # 确保 freq 数组不为空
            bins = freq2bin(freq, target_number)
            hist_file['bins'][i] = bins
        else:
            print(f"警告：第 {i} 行的 freq 数组为空，无法重新分箱")
            hist_file['bins'][i] = []  # 如果 freq 为空，设置为空列表
    return hist_file

def freq2bin(freqs, target_number):
    if len(freqs) == 0:
        return []

    freq = freqs.copy()
    maxi = len(freq) - 1

    step = 1. / target_number
    mini = 0
    while freq[mini + 1] == 0:
        mini += 1
    pointer = mini + 1
    cur_sum = 0
    res_pos = [mini]
    residue = 0
    while pointer < maxi + 1:
        cur_sum += freq[pointer]
        freq[pointer] = 0
        if cur_sum >= step:
            cur_sum -= step
            res_pos.append(pointer)
        else:
            pointer += 1

    if len(res_pos) == target_number:
        res_pos.append(maxi)

    return res_pos


class Batch():
    def __init__(self, attn_bias, rel_pos, heights, x, y=None):
        super(Batch, self).__init__()

        self.heights = heights
        self.x, self.y = x, y
        self.attn_bias = attn_bias
        self.rel_pos = rel_pos

    def to(self, device):
        self.heights = self.heights.to(device)
        self.x = self.x.to(device)
        self.attn_bias, self.rel_pos = self.attn_bias.to(device), self.rel_pos.to(device)
        return self

    def __len__(self):
        return self.in_degree.size(0)

def pad_1d_unsqueeze(x, padlen):
    x = x + 1  # pad id = 0
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen], dtype=x.dtype)
        new_x[:xlen] = x
        x = new_x
    return x.unsqueeze(0)

def pad_2d_unsqueeze(x, padlen):
    xlen, xdim = x.size()
    if xlen < padlen:
        new_x = x.new_zeros([padlen, xdim], dtype=x.dtype) + 1
        new_x[:xlen, :] = x
        x = new_x
    return x.unsqueeze(0)

def pad_rel_pos_unsqueeze(x, padlen):
    x = x + 1
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen], dtype=x.dtype)
        new_x[:xlen, :xlen] = x
        x = new_x
    return x.unsqueeze(0)

def pad_attn_bias_unsqueeze(x, padlen):
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen], dtype=x.dtype).fill_(float('-inf'))
        new_x[:xlen, :xlen] = x
        new_x[xlen:, :xlen] = 0
        x = new_x
    return x.unsqueeze(0)

def collator(small_set):
    y = small_set[1]
    xs = [s['x'] for s in small_set[0]]

    num_graph = len(y)
    x = torch.cat(xs)
    attn_bias = torch.cat([s['attn_bias'] for s in small_set[0]])
    rel_pos = torch.cat([s['rel_pos'] for s in small_set[0]])
    heights = torch.cat([s['heights'] for s in small_set[0]])

    return Batch(attn_bias, rel_pos, heights, x), y

def filterDict2Hist(hist_file, filterDict, encoding):
    buckets = len(hist_file['bins'].iloc[0])
    empty = np.zeros(buckets - 1)
    ress = np.zeros((3, buckets - 1))
    
    for i in range(len(filterDict['colId'])):
        colId = filterDict['colId'][i]
        col = encoding.idx2col[colId]
        
        if col == 'NA':
            ress[i] = empty
            continue
        
        matching_bins = hist_file.loc[hist_file['table_column'] == col, 'bins']
        
        if matching_bins.empty:
            print(f"警告：在直方图中未找到列 {col}")
            ress[i] = empty
            continue
        
        if len(matching_bins) > 1:
            print(f"警告：找到多个匹配的直方图条目，对于列 {col}，将使用第一个。")
        
        bins = matching_bins.iloc[0]

        opId = filterDict['opId'][i]
        op = encoding.idx2op[opId]

        val = filterDict['val'][i]
        mini, maxi = encoding.column_min_max_vals.get(col, (0, 1))
        
        # 检查 mini 和 maxi 是否是日期类型
        if isinstance(mini, datetime.date):
            print(f"mini 是日期类型：{mini}")
            mini = mini.toordinal()  # 将日期转换为序列化整数
        if isinstance(maxi, datetime.date):
            print(f"maxi 是日期类型：{maxi}")
            maxi = maxi.toordinal()  # 将日期转换为序列化整数

        # 确保 mini 和 maxi 是 float 类型
        mini = float(mini)
        maxi = float(maxi)

        val_unnorm = val * (maxi - mini) + mini

        left = 0
        right = len(bins) - 1
        
        # 找到 val_unnorm 在 bins 中的位置
        for j in range(len(bins)):
            if bins[j] < val_unnorm:
                left = j
            if bins[j] > val_unnorm:
                right = j
                break

        res = np.zeros(len(bins) - 1)

        # 根据操作符填充结果
        if op == '=':
            res[left:right] = 1
        elif op == '<':
            res[:left] = 1
        elif op == '>':
            res[right:] = 1
        
        ress[i] = res  # 完成 ress[i] 的赋值

    return ress


def formatJoin(json_node):
    join = None
    if 'Hash Cond' in json_node:
        join = json_node['Hash Cond']
    elif 'Join Filter' in json_node:
        join = json_node['Join Filter']
    elif 'Index Cond' in json_node and not json_node['Index Cond'][-2].isnumeric():
        join = json_node['Index Cond']

    if join is not None:
        twoCol = join[1:-1].split(' = ')
        twoCol = [json_node['Alias'] + '.' + col if len(col.split('.')) == 1 else col for col in twoCol]
        join = ' = '.join(sorted(twoCol))

    return join

def formatFilter(plan):
    alias = None
    if 'Alias' in plan:
        alias = plan['Alias']
    else:
        pl = plan
        while 'parent' in pl:
            pl = pl['parent']
            if 'Alias' in pl:
                alias = pl['Alias']
                break

    filters = []
    if 'Filter' in plan:
        filters.append(plan['Filter'])
    if 'Index Cond' in plan and plan['Index Cond'][-2].isnumeric():
        filters.append(plan['Index Cond'])
    if 'Recheck Cond' in plan:
        filters.append(plan['Recheck Cond'])

    return filters, alias

class Encoding:
    def __init__(self, column_min_max_vals, col2idx, op2idx=None):
        self.column_min_max_vals = column_min_max_vals
        self.col2idx = col2idx

        # 更新 op2idx，包含更多的运算符
        if op2idx is None:
            self.op2idx = {'>': 0, '=': 1, '<': 2, 'NA': 3, '>=': 4, '<=': 5, '!=': 6}
        else:
            self.op2idx = op2idx

        # 构建 idx2op 的反向映射
        self.idx2op = {v: k for k, v in self.op2idx.items()}

        # 初始化其他编码字典
        self.type2idx = {}
        self.idx2type = {}
        self.join2idx = {}
        self.idx2join = {}

        self.table2idx = {'NA': 0}
        self.idx2table = {0: 'NA'}

        # 构建 idx2col 的反向映射
        self.idx2col = {v: k for k, v in self.col2idx.items()}

    def normalize_val(self, column, val, log=False):
        # 获取列的最小值和最大值
        mini, maxi = self.column_min_max_vals.get(column, (0, 1))
        # 检查 mini 和 maxi 是否为 datetime.date 类型
        if isinstance(mini, datetime.date) and isinstance(maxi, datetime.date):
            # 将日期转换为序数
            mini = mini.toordinal()
            maxi = maxi.toordinal()
            val = val.toordinal() if isinstance(val, datetime.date) else val
        else:
            # 将 mini 和 maxi 转换为 float 类型
            mini = float(mini)
            maxi = float(maxi)
        val_norm = 0.0
        if maxi > mini:
            val_norm = (val - mini) / (maxi - mini)
        return val_norm

    def encode_filters(self, filters=[], alias=None):
        if len(filters) == 0:
            return {
                'colId': [self.col2idx.get('NA', 0)],
                'opId': [self.op2idx.get('NA', 3)],
                'val': [0.0]
            }
        res = {'colId': [], 'opId': [], 'val': []}
        for filt in filters:
            # 去除括号
            filt = filt.replace('(', '').replace(')', '')
            fs = filt.split(' AND ')
            for f in fs:
                # 使用正则表达式解析过滤条件
                import re
                pattern = r"(\S+)\s*(=|!=|>|<|>=|<=)\s*(.*)"
                match = re.match(pattern, f.strip())
                if match:
                    col, op, num = match.groups()
                    column = f"{alias}.{col}" if alias else col

                    # 更新运算符映射
                    if op not in self.op2idx:
                        self.op2idx[op] = len(self.op2idx)
                        self.idx2op[self.op2idx[op]] = op

                    colId = self.col2idx.get(column, self.col2idx.get('NA', 0))
                    opId = self.op2idx[op]

                    # 清理数值，去除引号和类型转换
                    num_clean = num.strip()
                    if '::' in num_clean:
                        num_clean = num_clean.split('::')[0]
                    num_clean = num_clean.strip("'")
                    try:
                        # 检查是否是日期格式
                        if re.match(r"\d{4}-\d{2}-\d{2}", num_clean):
                            num_value = datetime.datetime.strptime(num_clean, "%Y-%m-%d").date()
                        else:
                            num_value = float(num_clean)
                    except ValueError:
                        print(f"警告：无法将值 '{num}' 转换为浮点数或日期，使用默认值 0.0")
                        num_value = 0.0

                    # 调用修改后的 normalize_val 方法
                    val_norm = self.normalize_val(column, num_value)
                    res['colId'].append(colId)
                    res['opId'].append(opId)
                    res['val'].append(val_norm)
                else:
                    print(f"警告：无法解析过滤条件 '{f}'，已跳过。")
                    continue
        return res

    def encode_join(self, join):
        if join not in self.join2idx:
            self.join2idx[join] = len(self.join2idx)
            self.idx2join[self.join2idx[join]] = join
        return self.join2idx[join]

    def encode_table(self, table):
        if table not in self.table2idx:
            self.table2idx[table] = len(self.table2idx)
            self.idx2table[self.table2idx[table]] = table
        return self.table2idx[table]

    def encode_type(self, nodeType):
        if nodeType not in self.type2idx:
            self.type2idx[nodeType] = len(self.type2idx)
            self.idx2type[self.type2idx[nodeType]] = nodeType
        return self.type2idx[nodeType]

class TreeNode:
    def __init__(self, nodeType, typeId, filt, card, join, join_str, filterDict):
        self.nodeType = nodeType
        self.typeId = typeId
        self.filter = filt

        self.table = 'NA'
        self.table_id = 0
        self.query_id = None  # 用于样本位图识别

        self.join = join
        self.join_str = join_str
        self.card = card  # 'Actual Rows'
        self.children = []
        self.rounds = 0

        self.filterDict = filterDict

        self.parent = None

        self.feature = None

    def addChild(self, treeNode):
        self.children.append(treeNode)

    def __str__(self):
        return '{} with {}, {}, {} children'.format(self.nodeType, self.filter, self.join_str, len(self.children))

    def __repr__(self):
        return self.__str__()

    @staticmethod
    def print_nested(node, indent=0):
        print('--' * indent + '{} with {} and {}, {} childs'.format(node.nodeType, node.filter, node.join_str, len(node.children)))
        for k in node.children:
            TreeNode.print_nested(k, indent + 1)
