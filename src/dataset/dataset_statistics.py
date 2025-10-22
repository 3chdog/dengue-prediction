import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

from county_query import COUNTY_FROM_TOWN, COUNTY_LIST

# ========== 設定區 ==========
# 請將 'your_file.csv' 替換成你的實際檔案名稱
csv_file = 'Dengue_Daily.csv'

# 統計週期設定：'monthly' 為每月統計，'yearly' 為每年統計
period = 'monthly'  # 可選擇 'monthly' 或 'yearly'
# ============================

# 讀取CSV檔案
df = pd.read_csv(csv_file, encoding='utf-8-sig')

# 移除境外移入的資料
idx_to_remove = []
for i in range(len(df['發病日'])):
    if df['是否境外移入'].iloc[i] == '否':
        continue
    idx_to_remove.append(i)
print(f"共有 {len(idx_to_remove)} 筆境外移入資料將被移除")
df = df.drop(idx_to_remove).reset_index(drop=True)
print(df['是否境外移入'].value_counts())

# 檢查 '發病日' 欄位的年份是否在合理範圍內
idx_to_remove = []
for i in range(len(df['發病日'])):
    year_num = str(df['發病日'].iloc[i][:4])
    if year_num not in [str(y) for y in range(1970, 2025)]:
        print(f"發病日 '{df['發病日'].iloc[i]}' 的年份 {year_num} 不在範圍內")
        idx_to_remove.append(i)
print(f"共有 {len(idx_to_remove)} 筆發病日年份不合理的資料將被移除")
df = df.drop(idx_to_remove).reset_index(drop=True)
print(f"總共剩下 {len(df)} 筆資料")

# 填補 感染縣市 欄位的空值
county = 0
county_exist = []
no_county_use_county = []
town = []
a = 0
no_live_county = 0
has_live_county = 0
town_of_no_live_county = []
for i in range(len(df['發病日'])):
    if pd.isna(df['感染縣市'].iloc[i]):
        if pd.isna(df['居住縣市'].iloc[i]):
            no_live_county += 1
            if df['居住鄉鎮'].iloc[i] in ['東區', '中區', '南區']:
                a += 1
            town_of_no_live_county.append(df['居住鄉鎮'].iloc[i])
        else:
            has_live_county += 1
        county += 1
    else:
        county_exist.append(df['感染縣市'].iloc[i])
print(f"感染縣市 欄位共有 {county} 筆空值資料，將以 居住縣市 欄位填補")
print(f"感染鄉鎮 欄位共有 {len(town)} 筆空值資料，將以 居住鄉鎮 欄位填補")
print(f"感染縣市 欄位已有 {len(county_exist)} 筆資料")
print(f"感染縣市 欄位空值中，有 {has_live_county} 筆可用 居住縣市 欄位填補")
print(f"感染縣市 欄位空值中，有 {no_live_county} 筆無法用 居住縣市 欄位填補")
print(f"居住縣市 欄位也為空值且 居住鄉鎮 欄位為 東區、中區、南區 的共有 {a} 筆資料")
print(f"這些 居住鄉鎮 欄位的資料有：{set(town_of_no_live_county)}")
# print(set(county_exist))
# print(COUNTY_LIST)

# for i in COUNTY_LIST:
#     if i not in set(county_exist):
#         print(f"感染縣市 欄位缺少 {i} 的資料")


# # 檢查 居住縣市 欄位是否有空值
# a = 0
# county = []
# town = []
# for i in range(len(df['發病日'])):
#     if pd.isna(df['居住縣市'].iloc[i]):
#         # if pd.isna(df['居住鄉鎮'].iloc[i]):
#         #     print(f"  居住鄉鎮 欄位也為空值")
#         town.append(df['居住鄉鎮'].iloc[i])
#         a += 1
#     else:
#         county.append(df['居住縣市'].iloc[i])
# print(f"居住縣市 欄位共有 {a} 筆空值資料")


# a = 0
# b = 0
# inf_county = []
# for i in range(len(df['發病日'])):
#     if pd.isna(df['居住縣市'].iloc[i]):
#         a += 1
#         if pd.isna(df['感染縣市'].iloc[i]):
#             # print(f"  感染縣市 欄位也為空值")
#             b += 1
# print(f"居住縣市 欄位為空值的資料(共 {a} 筆)中，有 {b} 筆 感染縣市 欄位也為空值")





# # 檢查 感染縣市、感染鄉鎮 欄位是否有空值
# idx_inf_county = []
# idx_inf_town = []
# for i in range(len(df['發病日'])):
#     if pd.isna(df['感染縣市'].iloc[i]):
#         idx_inf_county.append(i)
#     if pd.isna(df['感染鄉鎮'].iloc[i]):
#         idx_inf_town.append(i)
# print(f"感染縣市 欄位共有 {len(idx_inf_county)} 筆空值資料")
# print(f"感染鄉鎮 欄位共有 {len(idx_inf_town)} 筆空值資料")

# # 列印感染縣市、感染鄉鎮 欄位為空值的例子
# print("例子：")
# for i in range(3):
#     idx = idx_inf_county[i]
#     print(df.iloc[idx])
#     print("")

# for i in range(100, 103):
#     idx = idx_inf_town[i]
#     print(df.iloc[idx])
#     print("")

# 將發病日轉換為日期格式
df['發病日'] = pd.to_datetime(df['發病日'], errors='coerce')

# 移除發病日為空值的資料
df = df.dropna(subset=['發病日'])

# 根據設定提取時間週期
if period == 'monthly':
    df['時間週期'] = df['發病日'].dt.to_period('M')
    period_name = '年月'
    title = '每月感染人數統計'
    output_prefix = 'monthly'
    xlabel = '年月'
elif period == 'yearly':
    df['時間週期'] = df['發病日'].dt.to_period('Y')
    period_name = '年份'
    title = '每年感染人數統計'
    output_prefix = 'yearly'
    xlabel = '年份'
else:
    raise ValueError("period 參數必須是 'monthly' 或 'yearly'")

# 統計感染人數
time_counts = df.groupby('時間週期').size().reset_index(name='感染人數')

# 將Period轉換為字串以便顯示
time_counts[period_name] = time_counts['時間週期'].astype(str)
time_counts = time_counts.drop('時間週期', axis=1)

# 顯示統計結果
print(f"\n{title}：")
print("="*40)
# print(time_counts.to_string(index=False))
print("="*40)
print(f"\n總計：{time_counts['感染人數'].sum()} 人")

# # 繪製圖表
# plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP']
# plt.figure(figsize=(12, 6))
# plt.bar(range(len(time_counts)), time_counts['感染人數'], color='steelblue')
# plt.xlabel(xlabel)
# plt.ylabel('感染人數')
# plt.title(title)
# plt.xticks(range(len(time_counts)), time_counts[period_name], rotation=45, ha='right')
# plt.grid(axis='y', alpha=0.3)
# plt.tight_layout()

# # 儲存圖表
# output_png = f'{output_prefix}_infection_stats.png'
# plt.savefig(output_png, dpi=300, bbox_inches='tight')
# print(f"\n圖表已儲存為 '{output_png}'")
# plt.show()

# # 輸出到CSV檔案
# output_csv = f'{output_prefix}_infection_summary.csv'
# time_counts.to_csv(output_csv, index=False, encoding='utf-8-sig')
# print(f"統計結果已儲存為 '{output_csv}'")
