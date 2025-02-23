### 專題名稱：姿態辨識與骨架分類系統

#### 專題簡介
本專題旨在開發一個基於姿態辨識和骨架分類的系統，該系統可以實時監測用戶的姿態並進行分類，從而提供姿態矯正的建議。系統主要分為前端和後端兩部分，前端負責捕捉用戶的姿態數據並傳送至後端，後端則負責數據處理和姿態分類。

#### 系統架構
1. **前端**：
   - 使用 MediaPipe Pose 進行姿態偵測。
   - 使用 JavaScript 和 HTML5 Canvas 進行視覺化展示。
   - 通過 Fetch API 將姿態數據傳送至後端。

2. **後端**：
   - 使用 Flask 框架搭建後端服務。
   - 使用 PyTorch 訓練和部署姿態分類模型。
   - 使用 MySQL 資料庫存儲用戶數據和分類結果。

#### 主要功能
1. **姿態偵測**：
   - 使用 MediaPipe Pose 偵測人體關鍵點。
   - 實時顯示偵測結果。

2. **姿態分類**：
   - 使用預訓練的 PyTorch 模型對姿態進行分類。
   - 根據分類結果提供姿態矯正建議。

3. **數據存儲與管理**：
   - 使用 MySQL 資料庫存儲用戶數據和分類結果。
   - 提供數據可視化功能，展示用戶的姿態數據和分類結果。

#### 文件結構
```
/final_v1
├── app.py                   # Flask 後端主程式
├── train.py                 # 模型訓練程式
├── requirements.txt         # 依賴包列表
├── static
│   ├── css
│   │   └── style.css        # 前端樣式表
│   ├── js
│   │   ├── pose.js          # 前端主程式
│   │   └── jquery.min.js    # jQuery 庫
│   └── audio
│       └── alert.mp3        # 提示音效
├── templates
│   ├── index.html           # 登入/註冊頁面
│   └── pose.html            # 姿態偵測頁面
└── user_data                # 用戶數據存儲目錄
```

#### 安裝與使用
1. **環境配置**：
   - 安裝 Python 3.8 或以上版本。
   - 安裝 MySQL 資料庫。

2. **安裝依賴**：
   ```bash
   pip install -r requirements.txt
   ```

3. **啟動後端服務**：
   ```bash
   python app.py
   ```

4. **訪問前端頁面**：
   在瀏覽器中打開 `http://localhost:5000`，進行登入或註冊，然後進入姿態偵測頁面。

#### 注意事項
- 請確保 MySQL 資料庫已啟動並配置正確的連接參數。
- 使用 MediaPipe Pose 進行姿態偵測時，請確保攝像頭正常工作。