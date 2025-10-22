import pandas as pd
import random
from datetime import datetime

def get_temperature(date_str, county):
    """
    根據日期和縣市產生合理的溫度值（攝氏度）
    台灣溫度範圍：冬季約 15-25°C，夏季約 28-35°C
    
    Args:
        date_str: 日期字串，格式為 'YYYY/MM/DD'
        county: 縣市名稱
        
    Returns:
        float: 溫度值（攝氏度）
    """
    try:
        date = datetime.strptime(date_str, '%Y/%m/%d')
        month = date.month
    except:
        month = random.randint(1, 12)
    
    # 根據月份調整溫度範圍
    if month in [12, 1, 2]:  # 冬季
        temp_range = (15, 25)
    elif month in [3, 4, 5]:  # 春季
        temp_range = (20, 30)
    elif month in [6, 7, 8]:  # 夏季
        temp_range = (28, 35)
    else:  # 秋季 (9, 10, 11)
        temp_range = (22, 30)
    
    # 南部通常較熱，北部較涼
    if county in ['屏東縣', '高雄市', '台南市']:
        adjustment = random.uniform(0, 2)
    elif county in ['台北市', '新北市', '基隆市']:
        adjustment = random.uniform(-2, 0)
    else:
        adjustment = 0
    
    temperature = random.uniform(temp_range[0], temp_range[1]) + adjustment
    return round(temperature, 1)


def get_humidity(date_str, county):
    """
    根據日期和縣市產生合理的相對濕度值（%）
    台灣濕度範圍：通常在 60-85% 之間，雨季可達 85-95%
    
    Args:
        date_str: 日期字串，格式為 'YYYY/MM/DD'
        county: 縣市名稱
        
    Returns:
        float: 相對濕度（%）
    """
    try:
        date = datetime.strptime(date_str, '%Y/%m/%d')
        month = date.month
    except:
        month = random.randint(1, 12)
    
    # 根據月份調整濕度範圍（5-9月為雨季）
    if month in [5, 6, 7, 8, 9]:  # 雨季
        humidity_range = (75, 90)
    else:  # 乾季
        humidity_range = (60, 80)
    
    # 東部和北部較潮濕
    if county in ['宜蘭縣', '花蓮縣', '台東縣', '基隆市']:
        adjustment = random.uniform(2, 5)
    else:
        adjustment = 0
    
    humidity = random.uniform(humidity_range[0], humidity_range[1]) + adjustment
    # 確保濕度不超過 100%
    humidity = min(humidity, 98)
    return round(humidity, 1)


def add_weather_data(input_csv, output_csv):
    """
    讀取登革熱 CSV 檔案，加入溫度和濕度資料後輸出
    
    Args:
        input_csv: 輸入的 CSV 檔案路徑
        output_csv: 輸出的 CSV 檔案路徑
    """
    # 讀取 CSV 檔案
    df = pd.read_csv(input_csv, encoding='utf-8-sig')
    
    # 為每筆資料加上溫度和濕度
    temperatures = []
    humidities = []
    
    for idx, row in df.iterrows():
        date = row.get('發病日', '')
        tmp_county = random.choice(['宜蘭縣', '花蓮縣', '台東縣', '基隆市'])
        county = row.get('居住縣市', tmp_county)
        
        temp = get_temperature(date, county)
        humid = get_humidity(date, county)
        
        temperatures.append(temp)
        humidities.append(humid)
    
    # 加入新欄位
    df['溫度(°C)'] = temperatures
    df['濕度(%)'] = humidities
    
    # 輸出到新的 CSV 檔案
    df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    
    print(f"處理完成！共處理 {len(df)} 筆資料")
    print(f"溫度範圍：{min(temperatures):.1f}°C - {max(temperatures):.1f}°C")
    print(f"濕度範圍：{min(humidities):.1f}% - {max(humidities):.1f}%")
    print(f"結果已儲存至：{output_csv}")
    
    return df


# 主程式
if __name__ == "__main__":
    # 設定輸入和輸出檔案路徑
    input_file = "Dengue_Daily.csv"  # 請修改為你的檔案名稱
    output_file = "Dengue_Daily_with_weather.csv"
    
    # 執行資料處理
    result_df = add_weather_data(input_file, output_file)
    
    # 顯示前幾筆資料作為範例
    print("\n前5筆資料預覽：")
    print(result_df[['發病日', '居住縣市', '溫度(°C)', '濕度(%)']].head())
