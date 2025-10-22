from playwright.sync_api import sync_playwright
from datetime import datetime, timedelta
import time
import os
from tqdm import tqdm

class TqdmWrapper(tqdm):
    """提供了一個 `total_time` 格式參數"""
    
    @property
    def format_dict(self):
        d = super().format_dict
        total_time = d["elapsed"] * (d["total"] or 0) / max(d["n"], 1)
        d.update(total_time='總計: ' + self.format_interval(total_time))
        return d

# 定義站名
station_name = "西屯 (C0F9T0)"
station_code = station_name.split()[-1].strip("()")

# 定義年份區間 (改為年份)
start_year = 2024
end_year = 2024  # 先測試下載2015年,成功後可改為 2024 下載多年

# 基本下載路徑
base_download_path = "./weather_data"

# 建立下載路徑
station_download_path = os.path.join(base_download_path, station_code)

if not os.path.exists(station_download_path):
    os.makedirs(station_download_path)

download_path = station_download_path

# 計算年份數量
years_list = list(range(start_year, end_year + 1))

def run(playwright):
    browser = playwright.chromium.launch(headless=True)  # 保持 False 以便觀察
    print('browser open')
    context = browser.new_context(accept_downloads=True)
    page = context.new_page()
    page.goto("https://codis.cwa.gov.tw/StationData")
    page.wait_for_load_state("load")
    
    # 選擇自動氣象站
    page.get_by_label("自動氣象站").check()
    time.sleep(1)
    
    # 輸入站名
    page.locator("li").filter(has_text="站名站號").get_by_role("combobox").click()
    page.locator("li").filter(has_text="站名站號").get_by_role("combobox").fill(station_name)
    page.locator(".leaflet-marker-icon > .icon_container > .marker_bgcolor > .bg_triangle").first.click()
    time.sleep(1)
    
    # 點擊資料圖表展示
    page.get_by_role("button", name="資料圖表展示").click()
    print("等待彈出視窗載入...")
    time.sleep(3)
    
    # 切換到年報表(逐月資料)
    print("尋找年報表選項...")
    try:
        page.get_by_text("年報表(逐月資料)").click()
        # year_report = page.locator("label").filter(has_text="年報表")
        # if year_report.count() > 0:
        #     print(f"找到 {year_report.count()} 個年報表選項,點擊第一個")
        #     year_report.first.click()
        # else:
        #     raise Exception("找不到年報表選項")
    except Exception as e:
        print(f"方法1失敗: {e}")
        page.get_by_text("年報表(逐月資料)").click()
    
    print("已切換到年報表模式")
    time.sleep(2)
    
    # 處理下載
    for year in TqdmWrapper(years_list, desc="下載進度", ncols=200, unit='file'):
        # 構建預期的檔名
        expected_filename = f"{station_code}-{year}.csv"
        expected_filepath = os.path.join(download_path, expected_filename)
        
        # 檢查檔案是否存在
        if os.path.exists(expected_filepath):
            print(f"\r檔案 {expected_filename} 已存在,跳過下載。", end=" ")
            continue
        
        # 選擇年份
        print(f"\r正在處理年份: {year}", end=" ")
        
        # ===== 修改部分開始 =====
        # 方法1: 使用 placeholder 精確定位第一個「請選擇年分」輸入框
        try:
            try:
                date_input = page.get_by_placeholder("2025").first
            except:
                date_input = page.get_by_placeholder("2025").first
            date_input.click()
            print(f"\r已點擊日期選擇器", end=" ")
            time.sleep(1)
            
            # 等待年份選擇器彈窗出現
            page.wait_for_selector(".vdatetime-year-picker", timeout=5000)
            
            # 尋找目標年份並點擊
            year_item = page.locator(f".vdatetime-year-picker__item").filter(has_text=str(year)).first
            
            if year_item.count() > 0:
                # 滾動到目標年份
                year_item.scroll_into_view_if_needed()
                time.sleep(0.5)
                # 點擊年份
                year_item.click()
                print(f"\r已選擇年份: {year}", end=" ")
                time.sleep(1)
                
                # 點擊 Continue 按鈕
                continue_btn = page.locator(".vdatetime-popup__actions__button--confirm")
                if continue_btn.count() > 0:
                    continue_btn.click()
                    print(f"\r已確認年份選擇", end=" ")
                    time.sleep(1)
            else:
                print(f"\r找不到年份 {year},跳過", end=" ")
                # 關閉彈出視窗
                page.keyboard.press("Escape")
                time.sleep(0.5)
                continue
                
        except Exception as e:
            print(f"\r選擇年份時發生錯誤: {e}", end=" ")
            # 嘗試方法2: 使用 datetime-tool-year 類別
            try:
                print(f"\r嘗試方法2...", end=" ")
                date_tool = page.locator(".datetime-tool-year").first
                date_tool.click()
                time.sleep(1)
                
                # 等待並選擇年份
                page.wait_for_selector(".vdatetime-year-picker", timeout=5000)
                year_item = page.locator(f".vdatetime-year-picker__item").filter(has_text=str(year)).first
                
                if year_item.count() > 0:
                    year_item.scroll_into_view_if_needed()
                    time.sleep(0.5)
                    year_item.click()
                    time.sleep(1)
                    
                    continue_btn = page.locator(".vdatetime-popup__actions__button--confirm")
                    if continue_btn.count() > 0:
                        continue_btn.click()
                        time.sleep(1)
                else:
                    print(f"\r找不到年份 {year},跳過", end=" ")
                    page.keyboard.press("Escape")
                    time.sleep(0.5)
                    continue
                    
            except Exception as e2:
                print(f"\r方法2也失敗: {e2}", end=" ")
                page.keyboard.press("Escape")
                time.sleep(0.5)
                continue
        # ===== 修改部分結束 =====
        
        # 執行下載操作
        print(f"\r正在下載: {expected_filename}", end=" ")
        time.sleep(1)
        
        with page.expect_download() as download_info:
            # 點擊 CSV 下載按鈕
            page.locator(".lightbox-tool-type-ctrl-btn-group > div").first.click()
            download = download_info.value
            download.save_as(os.path.join(download_path, download.suggested_filename))
            print("\r" + "檔案下載完成: " + download.suggested_filename, end=" ")
            time.sleep(2)
    
    print("\n所有檔案下載完成!")
    context.close()
    browser.close()

with sync_playwright() as playwright:
    run(playwright)

# ----------------------資料整理----------------------
print(f"資料路徑: {station_download_path}")

import pandas as pd
import glob
from prettytable import PrettyTable

def print_df_as_table(df, max_rows=5):
    table = PrettyTable()
    table.field_names = df.columns.tolist()
    for row in df.head(max_rows).itertuples(index=False):
        table.add_row(row)
    print(table)

# 匯入所有csv檔案
os.chdir(station_download_path)
extension = 'csv'
all_filenames = [i for i in glob.glob('*.{}'.format(extension))]

if all_filenames:
    # 讀取所有檔案,跳過第二行
    combined_csv = pd.concat([pd.read_csv(f, skiprows=[1]).assign(檔名=os.path.basename(f)) for f in all_filenames])
    
    # 提取檔名中的年份
    combined_csv['年份'] = combined_csv['檔名'].str.extract('(\d{4})')
    
    # 刪除檔名欄位
    combined_csv = combined_csv.drop(['檔名'], axis=1)
    
    print("\n合併後的資料預覽:")
    print_df_as_table(combined_csv, 10)
    
    # 儲存合併結果
    combined_csv.to_csv("combined_csv.csv", index=False, encoding='utf-8-sig')
    print(f"\n資料已儲存至: {os.path.join(station_download_path, 'combined_csv.csv')}")
else:
    print("沒有找到任何CSV檔案")
