import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 設定中文字體（如果需要顯示中文圖表）
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False

# 設定隨機種子
torch.manual_seed(42)
np.random.seed(42)


def preprocess_data(csv_file):
    """
    預處理登革熱資料，按月份統計感染人數及氣象資料
    
    Args:
        csv_file: CSV 檔案路徑
        
    Returns:
        DataFrame: 按月份統計的資料
    """
    # 讀取資料
    df = pd.read_csv(csv_file, encoding='utf-8-sig')
    
    # 轉換日期格式
    df['發病日'] = pd.to_datetime(df['發病日'], format='%Y/%m/%d', errors='coerce')
    
    # 移除無效日期
    df = df.dropna(subset=['發病日'])
    
    # 建立年月欄位
    df['年月'] = df['發病日'].dt.to_period('M')
    
    # 按月份統計
    monthly_data = df.groupby('年月').agg({
        '確定病例數': 'sum',
        '溫度(°C)': 'mean',
        '濕度(%)': 'mean'
    }).reset_index()
    
    # 轉換年月為 datetime
    monthly_data['日期'] = monthly_data['年月'].dt.to_timestamp()
    monthly_data = monthly_data.sort_values('日期').reset_index(drop=True)
    
    print(f"資料範圍：{monthly_data['日期'].min()} 至 {monthly_data['日期'].max()}")
    print(f"共 {len(monthly_data)} 個月的資料")
    print(f"平均每月感染人數：{monthly_data['確定病例數'].mean():.2f}")
    
    return monthly_data


class TimeSeriesDataset(Dataset):
    """
    時間序列資料集類別
    """
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def create_sequences(data, lookback=12):
    """
    建立時間序列資料集
    
    Args:
        data: 標準化後的資料
        lookback: 回顧的時間步長（預設12個月）
        
    Returns:
        X, y: 訓練特徵和標籤
    """
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i-lookback:i])
        y.append(data[i, 0])  # 預測確定病例數（第一欄）
    
    return np.array(X), np.array(y)


class LSTMModel(nn.Module):
    """
    LSTM 預測模型
    """
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        """
        Args:
            input_size: 輸入特徵數量
            hidden_size: LSTM 隱藏層大小
            num_layers: LSTM 層數
            dropout: Dropout 比率
        """
        super(LSTMModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM 層
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Dropout 層
        self.dropout = nn.Dropout(dropout)
        
        # 全連接層
        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 1)
    
    def forward(self, x):
        # LSTM 層
        lstm_out, _ = self.lstm(x)
        
        # 取最後一個時間步的輸出
        last_output = lstm_out[:, -1, :]
        
        # 通過全連接層
        out = self.dropout(last_output)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        
        return out


def train_model(model, train_loader, val_loader, criterion, optimizer, 
                num_epochs=100, patience=10, device='cpu'):
    """
    訓練模型
    
    Args:
        model: LSTM 模型
        train_loader: 訓練資料載入器
        val_loader: 驗證資料載入器
        criterion: 損失函數
        optimizer: 優化器
        num_epochs: 訓練週期數
        patience: 早停耐心值
        device: 運算設備
        
    Returns:
        model: 訓練好的模型
        train_losses: 訓練損失歷史
        val_losses: 驗證損失歷史
    """
    model = model.to(device)
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # 訓練階段
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            # 前向傳播
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs.squeeze(), y_batch)
            
            # 反向傳播
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # 驗證階段
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs.squeeze(), y_batch)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        # 顯示進度
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], '
                  f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        # 早停檢查
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # 儲存最佳模型
            torch.save(model.state_dict(), 'best_lstm_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break
    
    # 載入最佳模型
    model.load_state_dict(torch.load('best_lstm_model.pth'))
    
    return model, train_losses, val_losses


def predict_next_month(model, last_sequence, scaler, device='cpu'):
    """
    預測下個月的感染人數
    
    Args:
        model: 訓練好的模型
        last_sequence: 最後的序列資料（已標準化）
        scaler: 標準化器
        device: 運算設備
        
    Returns:
        predicted_cases: 預測的感染人數
    """
    model.eval()
    with torch.no_grad():
        last_sequence_tensor = torch.FloatTensor(last_sequence).unsqueeze(0).to(device)
        prediction = model(last_sequence_tensor)
        
        # 反標準化（只針對確定病例數欄位）
        prediction_np = prediction.cpu().numpy()
        dummy = np.zeros((1, scaler.n_features_in_))
        dummy[:, 0] = prediction_np
        predicted_cases = scaler.inverse_transform(dummy)[:, 0][0]
    
    return max(0, predicted_cases)  # 確保預測值非負


def plot_results(train_losses, val_losses, y_true, y_pred, dates):
    """
    繪製訓練結果
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # 繪製損失曲線
    axes[0].plot(train_losses, label='Training Loss')
    axes[0].plot(val_losses, label='Validation Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # 繪製預測結果
    axes[1].plot(dates, y_true, label='Actual', marker='o')
    axes[1].plot(dates, y_pred, label='Predicted', marker='x')
    axes[1].set_xlabel('Date')
    axes[1].set_ylabel('Cases')
    axes[1].set_title('Dengue Fever Cases: Actual vs Predicted')
    axes[1].legend()
    axes[1].grid(True)
    axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('lstm_results.png', dpi=300, bbox_inches='tight')
    print("結果圖已儲存為 'lstm_results.png'")
    plt.show()


def main(csv_file, lookback=12, hidden_size=64, num_layers=2, 
         dropout=0.2, batch_size=16, num_epochs=100, learning_rate=0.001):
    """
    主函數
    
    Args:
        csv_file: 輸入的 CSV 檔案路徑
        lookback: 回顧時間步長（月數）
        hidden_size: LSTM 隱藏層大小
        num_layers: LSTM 層數
        dropout: Dropout 比率
        batch_size: 批次大小
        num_epochs: 訓練週期數
        learning_rate: 學習率
    """
    # 檢查是否有 GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用設備: {device}")
    
    # 1. 預處理資料
    print("\n=== 步驟 1: 資料預處理 ===")
    monthly_data = preprocess_data(csv_file)
    
    # 2. 準備特徵
    features = ['確定病例數', '溫度(°C)', '濕度(%)']
    data = monthly_data[features].values
    dates = monthly_data['日期'].values
    
    # 3. 標準化
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)
    
    # 4. 建立序列
    print("\n=== 步驟 2: 建立時間序列 ===")
    X, y = create_sequences(data_scaled, lookback)
    print(f"序列數量: {len(X)}")
    print(f"輸入形狀: {X.shape}, 輸出形狀: {y.shape}")
    
    # 5. 分割訓練集和測試集
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    test_dates = dates[lookback + train_size:]
    
    # 6. 建立資料載入器
    train_dataset = TimeSeriesDataset(X_train, y_train)
    test_dataset = TimeSeriesDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 7. 建立模型
    print("\n=== 步驟 3: 建立 LSTM 模型 ===")
    input_size = X.shape[2]
    model = LSTMModel(input_size, hidden_size, num_layers, dropout)
    print(f"模型參數數量: {sum(p.numel() for p in model.parameters())}")
    
    # 8. 定義損失函數和優化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 9. 訓練模型
    print("\n=== 步驟 4: 訓練模型 ===")
    model, train_losses, val_losses = train_model(
        model, train_loader, test_loader, criterion, optimizer,
        num_epochs=num_epochs, patience=15, device=device
    )
    
    # 10. 評估模型
    print("\n=== 步驟 5: 評估模型 ===")
    model.eval()
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            predictions.extend(outputs.cpu().numpy())
            actuals.extend(y_batch.numpy())
    
    # 反標準化
    predictions = np.array(predictions).reshape(-1, 1)
    actuals = np.array(actuals).reshape(-1, 1)
    
    dummy_pred = np.zeros((len(predictions), scaler.n_features_in_))
    dummy_pred[:, 0] = predictions.flatten()
    predictions_original = scaler.inverse_transform(dummy_pred)[:, 0]
    
    dummy_actual = np.zeros((len(actuals), scaler.n_features_in_))
    dummy_actual[:, 0] = actuals.flatten()
    actuals_original = scaler.inverse_transform(dummy_actual)[:, 0]
    
    # 計算評估指標
    mse = mean_squared_error(actuals_original, predictions_original)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actuals_original, predictions_original)
    r2 = r2_score(actuals_original, predictions_original)
    
    print(f"均方誤差 (MSE): {mse:.2f}")
    print(f"均方根誤差 (RMSE): {rmse:.2f}")
    print(f"平均絕對誤差 (MAE): {mae:.2f}")
    print(f"決定係數 (R²): {r2:.4f}")
    
    # 11. 預測下個月
    print("\n=== 步驟 6: 預測下個月感染人數 ===")
    last_sequence = data_scaled[-lookback:]
    next_month_cases = predict_next_month(model, last_sequence, scaler, device)
    
    # 修正日期計算
    last_date = pd.to_datetime(dates[-1])
    next_month_date = last_date + pd.DateOffset(months=1)
    
    print(f"預測日期: {next_month_date.strftime('%Y-%m')}")
    print(f"預測感染人數: {next_month_cases:.0f} 人")
    
    # 12. 繪製結果
    print("\n=== 步驟 7: 繪製結果 ===")
    plot_results(train_losses, val_losses, actuals_original, 
                predictions_original, test_dates)
    
    return model, scaler, next_month_cases


# 主程式執行
if __name__ == "__main__":
    # 設定檔案路徑
    csv_file = "Dengue_Daily_with_weather.csv"  # 請修改為你的檔案名稱
    
    # 執行訓練和預測
    model, scaler, prediction = main(
        csv_file=csv_file,
        lookback=12,           # 使用過去 12 個月的資料
        hidden_size=64,        # LSTM 隱藏層大小
        num_layers=2,          # LSTM 層數
        dropout=0.2,           # Dropout 比率
        batch_size=16,         # 批次大小
        num_epochs=100,        # 最大訓練週期
        learning_rate=0.001    # 學習率
    )
    
    print("\n訓練完成！模型已儲存為 'best_lstm_model.pth'")
