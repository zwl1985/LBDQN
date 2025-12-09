import pickle
import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm
import math


df = pd.read_csv('pre-data/foursquare/dataset_WWW_friendship_old.txt', sep='\t', header=None)
df.columns = ['source', 'target']
G = nx.from_pandas_edgelist(df, create_using=nx.DiGraph())


df_location = pd.read_csv('pre-data/foursquare/processed_foursquare.csv', sep=',')
df_location.columns = ['u', 'location', 'x', 'y']
df_location = df_location.drop_duplicates(subset=['u', 'location'])
print(df_location)

l = 20

locations1 = pd.read_csv(f'foursquare/foursquare_text_{l}.txt', sep='\t', header=None, names=['x', 'y', 'location'])
print(locations1)


user_loaction_count_dict = {}
location_user_dict = {}
counts = pd.read_csv('pre-data/foursquare/user_venue_checkin_counts.csv', sep=',')
print(counts.columns)

# 在调用函数前预计算位置签到次数
location_checkin_counts = counts.groupby('venue_id')['checkin_count'].sum()
location_checkin_dict = location_checkin_counts.to_dict()


# 创建字典方便查询
location_checkin_dict = location_checkin_counts.to_dict()
# 使用示例: print(location_checkin_dict['88c46bf20db295831bd2d1718ad7e6f5'])

counts = counts.to_numpy()
for line in tqdm(counts):
    user_id, location, count = line

    if int(user_id) not in user_loaction_count_dict:
        user_loaction_count_dict[int(user_id)] = {}
    user_loaction_count_dict[int(user_id)][location] = int(count)

    if location not in location_user_dict:
        location_user_dict[location] = []
    location_user_dict[location].append(int(user_id))


# 预处理：将df_location转换为更高效的数据结构
# 对重复的location取第一个出现的坐标
location_coords = df_location.drop_duplicates('location').set_index('location')[['x', 'y']].to_dict('index')

user_center_locations = {}
distances = []

with tqdm(total=len(user_loaction_count_dict)) as bar:
    for user, location_counts in user_loaction_count_dict.items():
        locations = list(location_counts.keys())
        
        # 批量获取坐标信息
        coords_data = []
        counts = []
        
        for loc in locations:
            if loc in location_coords:
                coord = location_coords[loc]
                coords_data.append([coord['x'], coord['y']])
                counts.append(location_counts[loc])
        
        if not coords_data:
            user_center_locations[user] = {'center': np.array([0, 0]), 'distance': 1e-10}
            distances.append(1e-5)
            bar.update(1)
            continue
            
        user_location_info = np.array(coords_data)  # [n_locations, 2]
        counts = np.array(counts, dtype=float)  # [n_locations]
        
        # 归一化权重
        if counts.sum() == 0:
            weights = np.full(len(counts), 1e-10)
        else:
            weights = counts / counts.sum()
        
        # 计算加权中心
        center = (user_location_info * weights[:, np.newaxis]).sum(axis=0)
        
        # 计算加权距离
        distances_from_center = np.sqrt(((user_location_info - center) ** 2).sum(axis=1))
        user_distance = (distances_from_center * weights).sum()
        
        user_center_locations[user] = {'center': center, 'distance': user_distance + 1e-10}
        distances.append(user_distance + 1e-10)
        
        bar.update(1)


def calculate_ps(R_u, R_v, d):
    """
    根据公式计算 pl
    """
    # 情况1: d >= R_u + R_v
    if d >= R_u + R_v:
        return 0
    
    # 情况2: |R_u - R_v| < d < R_u + R_v
    elif abs(R_u - R_v) < d < R_u + R_v:
        # 计算角度 theta_u 和 theta_v
        theta_u = math.acos((R_u**2 + d**2 - R_v**2) / (2 * R_u * d))
        theta_v = math.acos((R_v**2 + d**2 - R_u**2) / (2 * R_v * d))
        
        numerator = (R_u**2 * theta_u + R_v**2 * theta_v - 
                    R_u * d * math.sin(theta_u))
        denominator = 2 * math.pi * R_v**2
        
        return numerator / denominator
    
    # 情况3: R_u <= R_v 且 d <= R_v - R_u
    elif R_u <= R_v and d <= R_v - R_u:
        return R_u**2 / R_v**2
    
    # 情况4: R_u > R_v 且 d <= R_u - R_v
    elif R_u > R_v and d <= R_u - R_v:
        return R_v**2 / R_u**2
    
    else:
        return 0  # 默认值

def probability_phy_fast(user, locations, df_location):
    # 预处理：获取用户位置数据（不转换类型）
    user_locations = df_location[df_location['u'] == user][['x', 'y']].values
    
    if len(user_locations) == 0:
        return 0.01
    
    # 直接使用数组（不转换类型）
    locations_array = locations[['x', 'y']].values
    
    # 向量化计算所有距离
    diff = user_locations[:, np.newaxis, :] - locations_array[np.newaxis, :, :]
    distances = np.sqrt((diff ** 2).sum(axis=2))
    
    # 计算平均距离
    avg_distance = distances.mean()
    
    if avg_distance == 0:
        avg_distance = 1e-8
    
    # 计算计数
    count_user = len(user_locations)
    count_location = len(df_location[df_location['location'].isin(locations['location'].values)])
    
    # 计算概率
    B = 30
    F = B * (count_user * count_location) / (avg_distance ** 2)
    p = np.tanh(F)
    
    return p

# 然后在主函数中使用
def probability_phy_precomputed(user, locations, df_location, user_loaction_count_dict, location_checkin_dict):
    """
    修改后的函数，使用正确的计数方式：
    count_user: 用户在目标位置的签到次数总和
    count_location: 目标位置的总签到次数
    """
    # 获取目标位置列表
    target_locations = locations['location'].values
    
    # 计算用户在这些位置的签到次数总和
    count_user = 0
    if user in user_loaction_count_dict:
        user_locations = user_loaction_count_dict[user]
        # print(target_locations)
        # 使用向量化操作计算总和
        count_user = sum(user_locations.values())

    # 计算这些位置的总签到次数
    # 使用向量化操作直接求和
    count_location = sum(location_checkin_dict.get(loc, 0) for loc in target_locations)
    
    # count_user = max(count_user, 1)
    # count_location = max(count_location, 1)
    if count_user == 0 or count_location == 0:
        # print(1)
        return 0.001
    
    # 获取用户位置数据用于距离计算
    user_locations = df_location[df_location['u'] == user][['x', 'y']].values
    
    if len(user_locations) == 0:
        # print(2)
        return 0.001
    
    locations_array = locations[['x', 'y']].values
    
    # 距离计算
    diff = user_locations[:, np.newaxis, :] - locations_array[np.newaxis, :, :]
    distances = np.sqrt((diff ** 2).sum(axis=2))
    avg_distance = distances.mean()
    
    if avg_distance == 0:
        avg_distance = 1e-8
    
    # 计算概率
    B = 30
    F = B * (count_user * count_location) / (avg_distance ** 2)
    p = np.tanh(F)
    
    return max(p, 0.001) if not math.isnan(p) else 0.001

remove_edges = []
for u, v in tqdm(G.edges()):
    indegree_v = G.in_degree(v)
    
    if u not in user_center_locations or v not in user_center_locations:
        ps = 0
        lt_counts['none'] += 1
    else:
        # 获取用户u的中心坐标和半径
        x1, y1 = user_center_locations[u]['center']
        R_u = user_center_locations[u]['distance'] / 2  # 注意：这里可能需要调整半径的定义
        
        # 获取用户v的中心坐标和半径
        x2, y2 = user_center_locations[v]['center']
        R_v = user_center_locations[v]['distance'] / 2  # 注意：这里可能需要调整半径的定义
        
        # 计算两个中心点之间的距离
        d = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        
        # 使用公式计算 S_uv
        ps = calculate_ps(R_u, R_v, d)
        if ps < 0.002:
            lt_counts['else'] += 1
    
    # 设置边的属性
    G.edges[u, v]['ps'] = max(ps, 0.001)
    G.edges[u, v]['po'] = 1 / indegree_v
    G.edges[u, v]['ph'] = probability_phy_precomputed(
    u, locations1, df_location, 
    user_loaction_count_dict, location_checkin_dict)
    if math.isnan(G.edges[u, v]['ph']):
        remove_edges.append((u, v))
G.remove_edges_from(remove_edges)

print(len(remove_edges))


with open(f'foursquare/g_{l}.pkl', 'wb') as f:
    pickle.dump(G,f)
