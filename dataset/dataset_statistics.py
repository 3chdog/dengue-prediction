import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# ========== 設定區 ==========
# 請將 'your_file.csv' 替換成你的實際檔案名稱
csv_file = 'Dengue_Daily.csv'

# 統計週期設定：'monthly' 為每月統計，'yearly' 為每年統計
period = 'monthly'  # 可選擇 'monthly' 或 'yearly'
# ============================

# 讀取CSV檔案
df = pd.read_csv(csv_file, encoding='utf-8-sig')

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
print(time_counts.to_string(index=False))
print("="*40)
print(f"\n總計：{time_counts['感染人數'].sum()} 人")

# 繪製圖表
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP']
plt.figure(figsize=(12, 6))
plt.bar(range(len(time_counts)), time_counts['感染人數'], color='steelblue')
plt.xlabel(xlabel)
plt.ylabel('感染人數')
plt.title(title)
plt.xticks(range(len(time_counts)), time_counts[period_name], rotation=45, ha='right')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()

# 儲存圖表
output_png = f'{output_prefix}_infection_stats.png'
plt.savefig(output_png, dpi=300, bbox_inches='tight')
print(f"\n圖表已儲存為 '{output_png}'")
plt.show()

# 輸出到CSV檔案
output_csv = f'{output_prefix}_infection_summary.csv'
time_counts.to_csv(output_csv, index=False, encoding='utf-8-sig')
print(f"統計結果已儲存為 '{output_csv}'")
