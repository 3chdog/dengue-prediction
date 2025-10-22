from playwright.sync_api import sync_playwright
from datetime import datetime, timedelta
import time
import os
import re
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
    # for year in TqdmWrapper(years_list, desc="下載進度", ncols=200, unit='file'):
    for year in years_list:
        # 構建預期的檔名
        expected_filename = f"{station_code}-{year}.csv"
        expected_filepath = os.path.join(download_path, expected_filename)
        
        # 檢查檔案是否存在
        if os.path.exists(expected_filepath):
            print(f"\r檔案 {expected_filename} 已存在,跳過下載。", end=" ")
            time.sleep(1)
            continue
        
        # 選擇年份
        print(f"\r正在處理年份: {year}", end=" ")

        try:
            # # 1️⃣ 點擊「觀測時間」右邊的年選擇器
            # year_selectors = page.locator("div.datetime-tool.datetime-tool-year input.vdatetime-input")
            # print("偵測到年分選擇器數量:", year_selectors.count())
            # year_selector = None
            # for i in range(year_selectors.count()):
            #     try:
            #         year_selector = year_selectors.nth(i)
            #         text = year_selector.input_value()
            #         print(f"嘗試第 {i} 個選擇器: {text}")
            #         year_selector.click(timeout=5000)
            #         time.sleep(0.5)
            #         print(f"✅ 已點擊第 {i} 個選擇器")
            #         break
            #     except Exception as e:
            #         print(f"❌ 點擊第 {i} 個選擇器失敗，嘗試下一個。")
            #         # print(f"❌ 點擊第 {i} 個選擇器失敗: {e}")
  
            # 1️⃣ 點擊「觀測時間」右邊的年選擇器
            year_selector = page.locator("div.datetime-tool.datetime-tool-year input.vdatetime-input").nth(2)
            year_selector.click(timeout=10000)
            time.sleep(0.5)
            print("已點擊年份選擇器")

            
            # 2️⃣ 等待彈出視窗 (.vdatetime-popup) 顯示
            page.wait_for_selector(".vdatetime-popup", state="visible", timeout=5000)
            time.sleep(0.5)
            print("✅ 年份選擇彈窗已顯示")
            
            # 3️⃣ 找到對應年份（可能要滾動才能出現）
            year_item_locator = page.locator(f".vdatetime-year-picker__item >> text={year}")
            time.sleep(0.5)
            print("✅ 尋找年份項目定位器完成")
            
            # 滾動直到找到
            for _ in range(10):
                if year_item_locator.count() > 0:
                    break
                page.mouse.wheel(0, 200)  # 模擬滾動
                time.sleep(0.3)
            
            if year_item_locator.count() == 0:
                print(f"⚠️ 無法找到年份 {year}，跳過")
                page.keyboard.press("Escape")
                continue

            # 點擊年份
            year_item_locator.first.click()
            print(f"✅ 已點擊年份 {year}")

            # 4️⃣ 點擊 Continue 按鈕 (如果有的話)
            continue_btn = page.locator(".vdatetime-popup__actions__button--confirm", has_text="Continue")
            continue_btn.click()
            print("已確認年份")

            # 5️⃣ 等待彈窗關閉
            page.wait_for_selector(".vdatetime-popup", state="hidden", timeout=5000)
            time.sleep(0.5)

        except Exception as e:
            print(f"❌ 選擇年份失敗: {e}")
            page.keyboard.press("Escape")
            continue
        
        # 執行下載操作
        print(f"\r正在下載: {expected_filename}", end=" ")
        time.sleep(1)
        with page.expect_download() as download_info:
            # page.locator(".lightbox-tool-type-ctrl-btn-group > div").first.click()
            # page.wait_for_selector(".lightbox-tool-type-ctrl-btn-group > div", state="visible", timeout=10000)
            # page.locator(".lightbox-tool-type-ctrl-btn-group > div").first.click()
            # # download_btn = page.locator('div.lightbox-tool-type-ctrl-btn', has_text="CSV下載")
            # download_btn = page.locator('div.lightbox-tool-type-ctrl-btn', has_text=re.compile("CSV\s*下載"))
            # download_btn.wait_for(state="visible", timeout=10000)
            print("✅ 找到 CSV下載 按鈕，準備點擊")
            # download_btn.click()

            download_btn_selectors = page.locator("div.lightbox-tool-type-ctrl-btn", has_text="CSV下載")
            print("偵測到下載按鈕數量:", download_btn_selectors.count())
            btn_selector = None
            for i in range(download_btn_selectors.count()):
                try:
                    # 1️⃣ 找到該元素（用 innerText 匹配）
                    btn_selector = download_btn_selectors.nth(i)
                    text = btn_selector.inner_text()
                    print(f"嘗試第 {i} 個按鈕: {text}", end=" ")
                    btn_selector.click(timeout=5000)
                    time.sleep(0.5)
                    print(f"\n✅ 已點擊第 {i} 個按鈕")

                    # 2️⃣ 確保元素存在
                    btn_selector.wait_for(state="attached", timeout=5000)
                    print("✅ CSV下載按鈕已找到，下一步使用 JS 強制觸發 click()")

                    # 3️⃣ 透過 JS 點擊，不經由 Playwright 的 visibility 檢查
                    page.evaluate("(el) => el.click()", btn_selector.element_handle())
                    print("✅ 已透過 JS 強制觸發 click()")
                    break
                except Exception as e:
                    print(f"❌ 點擊第 {i} 個按鈕失敗，嘗試下一個。")
                    # print(f"❌ 點擊第 {i} 個按鈕失敗: {e}")





            # # 1️⃣ 找到該元素（用 innerText 匹配）
            # download_btn = page.locator("div.lightbox-tool-type-ctrl-btn", has_text="CSV下載")

            # # 2️⃣ 確保元素存在
            # download_btn.wait_for(state="attached", timeout=5000)
            # print("✅ CSV下載按鈕已找到，使用 JS 強制觸發 click()")

            # # 3️⃣ 透過 JS 點擊，不經由 Playwright 的 visibility 檢查
            # page.evaluate("(el) => el.click()", download_btn.element_handle())

            # # 4️⃣ 若網站自動下載，則這裡等待下載事件
            # with page.expect_download() as download_info:
            #     pass  # 讓 Playwright 監聽
            download = download_info.value
            download.save_as(download_path + "/" + download.suggested_filename)
            print(f"✅ 已下載: {download.suggested_filename}")






            # download = download_info.value
            # download.save_as(download_path+"/" +  download.suggested_filename)  # 儲存檔案
            # #print(download.url)  # 獲取下載的url地址
            # # 這一步只是下載下來，生成一個隨機uuid值儲存，程式碼執行完會自動清除
            # print("\r"+"檔案不存在，以下載 : "+download.suggested_filename,end=" ")  # 獲取下載的檔名
            # time.sleep(1)
            print("BEFORE END")
            # page.locator("div:nth-child(5) > .lightbox-tool-type-ctrl > .lightbox-tool-type-ctrl-form > label > .datetime-tool > div").first.click()
            # print("ENDDDING")
            # time.sleep(2)
    
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
