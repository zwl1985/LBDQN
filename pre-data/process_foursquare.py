# %%
import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm


# %%
checkins = pd.read_csv('dataset_WWW_Checkins_anonymized.txt', sep='\t', header=None, names=['user_id', 'venue_id', 'utc', 'timestamp'])

# %%
checkins = checkins[['user_id', 'venue_id']]    

# %%
checkins

# %%
raw_checkins = pd.read_csv('foursquare/raw_POIs.txt', sep='\t', header=None, names=['venue_id', 'Longitude', 'Latitude', 'Category', 'Country'])

# %%
raw_checkins = raw_checkins[['venue_id', 'Longitude', 'Latitude']]

# %%
# 使用 merge 进行列合并，以 checkins 为主表，根据 venue_id 进行左连接
merged_data = checkins.merge(raw_checkins, on='venue_id', how='left')

# %%
merged_data

# %%
# 保存合并后的数据到 CSV 文件
merged_data.to_csv('processed_foursquare.csv', index=False)

# %%
# 另一种方式：使用 value_counts
checkin_counts = merged_data[['user_id', 'venue_id']].value_counts().reset_index(name='checkin_count')

# %%
checkin_counts

# %%
# 保存结果到CSV文件
checkin_counts.to_csv('user_venue_checkin_counts.csv', index=False)

# %%
for l in [1, 5, 10, 15, 20]:
    # 从checkin_counts中选择l个场所ID
    rows = np.arange(l)
    selected_venue_ids = checkin_counts[['venue_id']].iloc[rows]
    
    # 从merged_data中提取这些场所ID对应的经纬度信息
    venue_info = merged_data[['Longitude', 'Latitude', 'venue_id']]
    selected_venues = venue_info[venue_info['venue_id'].isin(selected_venue_ids['venue_id'])]
    
    # 去重（因为一个场所可能被多个用户签到过）
    selected_venues = selected_venues.drop_duplicates(subset=['venue_id'])
    
    # 保存到文件
    selected_venues[['Longitude', 'Latitude', 'venue_id']].to_csv(f'../MLBSN/foursquare/foursquare_text_{l}.txt', 
                                                                  index=False, header=False, sep='\t')

# %%
merged_data[['Longitude', 'Latitude', 'venue_id']].iloc[[2, 5 ,3]]

# %%


# %%



