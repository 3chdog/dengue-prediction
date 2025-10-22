import pandas as pd

from county_query import COUNTY_LIST

# 取得有值與無值的索引
def get_indexes_split_by_none(df: pd.DataFrame, column_name: str):
    idx_with_value = df[~pd.isna(df[column_name])].index.tolist()
    print(f"[{column_name}] 有值的部分共有 {len(idx_with_value)} 筆資料")
    idx_without_value = df[pd.isna(df[column_name])].index.tolist()
    print(f"[{column_name}] 無值的部分共有 {len(idx_without_value)} 筆資料")
    assert len(idx_with_value) + len(idx_without_value) == len(df), f"索引數量不匹配，總數量應為 {len(df)}"
    return idx_with_value, idx_without_value

# 移除境外移入的資料
def remove_travelers_entering_samples(df: pd.DataFrame) -> pd.DataFrame:
    idx_to_remove = []
    for i in range(len(df['發病日'])):
        if df['是否境外移入'].iloc[i] == '否':
            continue
        idx_to_remove.append(i)
    print(f"[境外移入] 共有 {len(idx_to_remove)} 筆境外移入資料將被移除")
    df = df.drop(idx_to_remove).reset_index(drop=True)
    print(f"總共剩下 {len(df)} 筆資料")
    return df

# 檢查 '發病日' 欄位的年份是否在合理範圍內
def check_reported_date_range(df: pd.DataFrame, start_year: int, end_year: int) -> pd.DataFrame:
    idx_to_remove = []
    for i in range(len(df['發病日'])):
        year_num = str(df['發病日'].iloc[i][:4])
        if year_num not in [str(y) for y in range(start_year, end_year + 1)]:
            # print(f"發病日 '{df['發病日'].iloc[i]}' 的年份 {year_num} 不在範圍內")
            idx_to_remove.append(i)
    print(f"[日期範圍] 共有 {len(idx_to_remove)} 筆發病日年份不合理的資料將被移除")
    df = df.drop(idx_to_remove).reset_index(drop=True)
    print(f"總共剩下 {len(df)} 筆資料")
    return df

# 檢查樣本中的縣市是否在 COUNTY_LIST 中
def check_samples_in_county_list(df: pd.DataFrame, idx_list: list, keyword: str = '感染縣市'):
    missing_counties = set()
    for i in idx_list:
        county = df[keyword].iloc[i]
        if county not in COUNTY_LIST:
            missing_counties.add(county)
    if missing_counties:
        print(f"[縣市檢查] 以下縣市不在 COUNTY_LIST 中：{missing_counties}")
    else:
        print(f"[縣市檢查] 所有樣本的縣市皆在 COUNTY_LIST 中")

# 主要的前處理函式
def dengue_preprocess(csv_file: str, period: str = 'monthly'):
    df = pd.read_csv(csv_file, encoding='utf-8-sig')
    df = remove_travelers_entering_samples(df)
    df = check_reported_date_range(df, 1998, 2024)

    # 剩下的樣本分成 '感染縣市'有值 與 '感染縣市'無值 兩部分
    idx_with_infection_county, idx_without_infection_county = get_indexes_split_by_none(df, '感染縣市')
    check_samples_in_county_list(df, idx_with_infection_county)

if __name__ == "__main__":
    csv_file = 'Dengue_Daily.csv'
    period = 'monthly' # 統計週期設定：'monthly' 為每月統計，'yearly' 為每年統計
    dengue_preprocess(csv_file, period)
