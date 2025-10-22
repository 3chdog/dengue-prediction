#!/usr/bin/env bash
# ==========================================
# 自動安裝 ChromeDriver 到使用者目錄 ~/bin
# 並自動更新 PATH
# 適用於 Linux / Ubuntu / WSL2
# ==========================================

set -e

echo "🚀 開始安裝 ChromeDriver..."

# 建立目錄
mkdir -p ~/bin
cd ~/bin

# 偵測是否安裝 Chrome 或 Chromium
if command -v google-chrome >/dev/null 2>&1; then
  CHROME_CMD="google-chrome"
elif command -v chromium-browser >/dev/null 2>&1; then
  CHROME_CMD="chromium-browser"
else
  echo "❌ 未偵測到 Google Chrome 或 Chromium，請先安裝。"
  echo "👉 例如：sudo apt install -y google-chrome-stable"
  exit 1
fi

# 取得 Chrome 版本號
CHROME_VERSION=$($CHROME_CMD --version | grep -oE "[0-9]+(\.[0-9]+)+" | head -n1)
MAJOR_VERSION=$(echo $CHROME_VERSION | cut -d '.' -f 1)

echo "🌐 偵測到 Chrome 版本：$CHROME_VERSION（主要版本 $MAJOR_VERSION）"

# 下載對應版本的 ChromeDriver
ZIP_URL="https://storage.googleapis.com/chrome-for-testing-public/${CHROME_VERSION}/linux64/chromedriver-linux64.zip"

echo "⬇️ 下載中：$ZIP_URL (下載至 ~/bin/chromedriver.zip)..."
wget -q "$ZIP_URL" -O chromedriver.zip

# 解壓縮
unzip -o chromedriver.zip
mv chromedriver-linux64/chromedriver ./
chmod +x chromedriver
rm -rf chromedriver-linux64 chromedriver.zip

echo "✅ ChromeDriver 已安裝至 ~/bin/chromedriver"
echo

# 更新 PATH
SHELL_RC=""
if [ -n "$ZSH_VERSION" ]; then
  SHELL_RC="$HOME/.zshrc"
elif [ -n "$BASH_VERSION" ]; then
  SHELL_RC="$HOME/.bashrc"
else
  SHELL_RC="$HOME/.profile"
fi

if ! grep -q 'export PATH="$HOME/bin:$PATH"' "$SHELL_RC"; then
  echo 'export PATH="$HOME/bin:$PATH"' >> "$SHELL_RC"
  echo "🔧 已將 ~/bin 加入 PATH（修改於 $SHELL_RC）"
else
  echo "ℹ️ 你的 PATH 已包含 ~/bin，略過此步。"
fi

# 重新加載設定
source "$SHELL_RC"

echo
echo "🎉 安裝完成！請重新開啟終端機或執行以下命令以生效："
echo "source $SHELL_RC"
echo "(可能需要重啟conda環境)"
echo
echo "🔍 執行以下命令以驗證："
echo "which chromedriver"
echo "chromedriver --version"
echo
echo "若以上命令成功執行，則表示 ChromeDriver 安裝成功！可接著使用query_weather.py進行天氣資料抓取。"
