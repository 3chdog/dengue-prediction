import pandas as pd

from .county_query import COUNTY_LIST, COUNTY_FROM_TOWN

# 取得有值與無值的索引
def get_indexes_split_by_none(df: pd.DataFrame, column_name: str) -> tuple[list[int], list[int]]:
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

# 檢查索引是否有重複
def check_indexes_no_duplicate(list_of_indexes: list[list[int]]) -> bool:
    list_of_sets = [set(idx_list) for idx_list in list_of_indexes]
    is_duplicate = False
    for one_set in list_of_sets:
        others = [s for s in list_of_sets if s != one_set]
        union_of_others = set().union(*others)
        if one_set & union_of_others:
            is_duplicate = True
            print(f"發現重複的索引：{one_set & union_of_others}")
    if not is_duplicate:
        total_count = sum([len(idx_list) for idx_list in list_of_indexes])
        print(f"所有索引皆無重複，總計 {total_count} 筆資料")
    return is_duplicate

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

# 填值：利用 '來源欄位' 填補 '目標欄位' 的值 (支援'來源欄位'的值的轉換)
def fill_values(df: pd.DataFrame, target_indexes: list[int], target_column: str, input_column: str, transform_dict: dict = None, verbose: bool = False) -> pd.DataFrame:
    for i in target_indexes:
        # 檢查目標欄位
        if not pd.isna(df[target_column].iloc[i]) and verbose:
            print(f"[填值警告] 索引 {i} 的 '{target_column}' 欄位已有值")

        # 檢查來源欄位
        if pd.isna(df[input_column].iloc[i]):
            if verbose:
                print(f"[填值跳過] 索引 {i} 的 '{input_column}' 欄位無值，無法填補 '{target_column}' 欄位")
            continue

        # 進行轉換（如果有提供轉換字典）
        input_value = df[input_column].iloc[i]
        if transform_dict and input_value in transform_dict:
            input_value = transform_dict[input_value]
        df.at[i, target_column] = input_value

        # 確認填值成功
        if verbose:
            print(f"[填值成功] 索引 {i} 的 '{target_column}' 欄位已填補為 '{input_value}'")
        assert df.iloc[i][target_column] is not None, f"[填值錯誤] 索引 {i} 的 '{target_column}' 欄位填補失敗"
    print(f"[填值完成] 共填補 {len(target_indexes)} 筆資料，將 '{input_column}' 值填於 '{target_column}' 欄位")
    return df

# 主要的前處理函式
def dengue_preprocess(csv_file: str, period: str = 'monthly'):
    """
     1. 移除境外移入的資料
     2. 檢查發病日年份範圍
     3. 分析並填補感染縣市欄位的空值
     4. 最終檢查保留欄位是否有空值
    """
    df = pd.read_csv(csv_file, encoding='utf-8-sig')
    df = remove_travelers_entering_samples(df)
    df = check_reported_date_range(df, 1998, 2024)

    # 剩下的樣本分成 '感染縣市'有值 與 '感染縣市'無值 兩部分
    idx_with_infection_county, idx_without_infection_county = get_indexes_split_by_none(df, '感染縣市')
    check_samples_in_county_list(df, idx_with_infection_county)

    # 將 '感染縣市'無值 的部分分成 '居住縣市'有值 與 '居住縣市'無值 兩部分
    df_no_infection_county = df.loc[idx_without_infection_county]
    (
        idx_without_infection_county_but_living_county,
        idx_without_infection_county_neither_living_county
    ) = get_indexes_split_by_none(df_no_infection_county, '居住縣市')

    # 檢查目前分成三群的索引是否有重複
    check_indexes_no_duplicate(
        [
            idx_with_infection_county,
            idx_without_infection_county_but_living_county,
            idx_without_infection_county_neither_living_county,
        ]
    )

    # 利用 '居住縣市' 欄位填補 '感染縣市' 欄位的空值
    df = fill_values(
        df,
        idx_without_infection_county_but_living_county,
        target_column='感染縣市',
        input_column='居住縣市',
        transform_dict=None,
        # verbose=True
    )

    df = fill_values(
        df,
        idx_without_infection_county_neither_living_county,
        target_column='感染縣市',
        input_column='居住鄉鎮',
        transform_dict=COUNTY_FROM_TOWN,
        # verbose=True
    )

    # 保留必要的欄位
    columns_to_keep = ['發病日', '感染縣市']
    df = df[columns_to_keep]

    # 最後檢查保留欄位是否有空值
    for column in columns_to_keep:
        _, idx_without_value = get_indexes_split_by_none(df, column)
        assert len(idx_without_value) == 0, f"[最終檢查錯誤] 欄位 '{column}' 中仍有空值"

if __name__ == "__main__":
    csv_file = 'Dengue_Daily.csv'
    period = 'monthly' # 統計週期設定：'monthly' 為每月統計，'yearly' 為每年統計
    dengue_preprocess(csv_file, period)
