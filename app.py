# hint:
# 這邊的所有功能都是依照session['user_id']來區分
# 目前只有一個使用者，所以session['user_id'] = "i am user"
# 如果要改成多個使用者，可以在LOGING的時候，將session['user_id']設定為使用者的id
# 這樣就可以區分不同使用者的資料nb

from flask import Flask, request, jsonify, session, redirect, url_for, render_template
from flask import render_template
import numpy as np
import torch
import torch.nn as nn
import random
import train
import time
import os
import uuid
import shutil
import pymysql
# add
import json
import csv
from datetime import datetime, timedelta
import pandas as pd
import plotly.graph_objects as go
from plotly_resampler import FigureResampler
from scipy.ndimage import gaussian_filter1d
app = Flask(__name__)
app.secret_key = 'supersecretkey'
# Define binary classification model
class PostureClassifier(nn.Module):
    def __init__(self, input_dim):
        super(PostureClassifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # Output probability between 0-1
        )

    def forward(self, x):
        return self.fc(x)

model_dic = {'default': PostureClassifier(39)}




### 環境相關
@app.route('/init', methods=['POST'])
def init():
    global model_dic
    if 'user_id' not in session:
        return jsonify(error='User not logged in')
    session['recording_normal'] = False
    session['recording_abnormal'] = False
    # 這邊的所有功能都是依照session['user_id']來區分
    #session["user_id"] = "i am user"
    # 如果要改成多個使用者，可以在LOGING的時候，將session['user_id']設定為使用者的id，然後刪掉上面這行

    if os.path.exists('user_data/' + session['user_id'] + '/model.pth'):
        model = PostureClassifier(39)
        model.load_state_dict(torch.load('user_data/' + session['user_id'] + '/model.pth'))
        with open('user_data/' + session['user_id'] + '/model_info.txt', 'r') as f:
            model_info = f.read()
            session['model_info']=model_info
        model_dic[session['user_id']] = model
    #統計normal和abnormal的數量
    return jsonify(success=True)

@app.route('/reset_env', methods=['POST'])
def reset_env():
    global model_dic
    session['recording_normal'] = False
    session['recording_abnormal'] = False
    shutil.rmtree('user_data/' + session['user_id'])
    #統計normal和abnormal的數量
    return jsonify(success=True)

### 資料獲取相關
@app.route('/get_file_counts', methods=['POST'])
def get_file_counts():
    normal_num = 0
    abnormal_num = 0
    if(os.path.exists('user_data/' + session['user_id'] + '/normal')):
        normal_num = len(os.listdir('user_data/' + session['user_id'] + '/normal'))
    if(os.path.exists('user_data/' + session['user_id'] + '/abnormal')):
        abnormal_num = len(os.listdir('user_data/' + session['user_id'] + '/abnormal'))
    return jsonify(success=True, normal=normal_num, abnormal=abnormal_num)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    landmarks = data.get('landmarks', [])
    if not landmarks:
        return jsonify({'error': 'No landmarks provided'})
    ref_landmark = landmarks[11]  # Reference point
    ref_x, ref_y, ref_z = ref_landmark['x'], ref_landmark['y'], ref_landmark['z']

    # Calculate relative positions
    feature_vector = []
    for lm in landmarks[:13]:  # Only take the first 13 landmarks
        feature_vector.extend([lm['x'] - ref_x, lm['y'] - ref_y, lm['z'] - ref_z])
    feature_vector = np.array(feature_vector, dtype=np.float32)
    if session.get('recording_normal', False):
        os.makedirs('user_data/' + session['user_id'] + '/normal', exist_ok=True)
        data_folder = 'user_data/' + session['user_id'] + '/normal'
        np.save(data_folder + '/' + str(time.time()) + '.npy', feature_vector)
    
    if session.get('recording_abnormal', False):
        os.makedirs('user_data/' + session['user_id'] + '/abnormal', exist_ok=True)
        data_folder = 'user_data/' + session['user_id'] + '/abnormal'
        np.save(data_folder + '/' + str(time.time()) + '.npy', feature_vector)
    
    # 回傳預測結果
    if os.path.exists('user_data/' + session['user_id'] + '/model.pth') == False:
        model = PostureClassifier(39)
        model_dic[session['user_id']] = model
        session['model_info'] = 'default'
    else:
        with open('user_data/' + session['user_id'] + '/model_info.txt', 'r') as f:
            model_info = f.read()
        if(model_info != session['model_info']):
            model = PostureClassifier(39)
            model.load_state_dict(torch.load('user_data/' + session['user_id'] + '/model.pth'))
            model_dic[session['user_id']] = model
            session['model_info'] = model_info
    model = model_dic.get(session['user_id'], model_dic['default'])
    feature_tensor = torch.tensor(feature_vector).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        prediction = model(feature_tensor).item()
    # add
    classification = 'Good' if prediction > 0.5 else 'Bad'
    save_classification_result(session['user_id'], prediction, classification)


    return jsonify({'prediction': prediction})



### 控制ui相關
@app.route('/start_recording_normal', methods=['POST'])
def start_recording_normal():
    print("start_recording_normal")
    session['recording_normal'] = True
    return jsonify({'message': 'Recording normal posture'})

@app.route('/stop_recording_normal', methods=['POST'])
def stop_recording_normal():
    session['recording_normal'] = False
    return jsonify({'message': 'Stopped recording normal posture'})

@app.route('/start_recording_abnormal', methods=['POST'])
def start_recording_abnormal():
    session['recording_abnormal'] = True
    return jsonify({'message': 'Recording abnormal posture'})

@app.route('/stop_recording_abnormal', methods=['POST'])
def stop_recording_abnormal():
    session['recording_abnormal'] = False
    return jsonify({'message': 'Stopped recording abnormal posture'})
@app.route('/start_training', methods=['POST'])
def start_training():
    user_id = session['user_id']
    normal_data = []
    abnormal_data = []
    
    for file in os.listdir('user_data/' + user_id + '/normal'):
        normal_data.append(np.load(os.path.join('user_data/' + user_id + '/normal', file)))
    
    for file in os.listdir('user_data/' + user_id + '/abnormal'):
        abnormal_data.append(np.load(os.path.join('user_data/' + user_id + '/abnormal', file)))

    acc, model = train.train_model(normal_data, abnormal_data)  # Pass the device to training function
    # model_dic[user_id] = model
    
    # Save the model
    torch.save(model.state_dict(), 'user_data/' + user_id + '/model.pth')
    with open('user_data/' + user_id + '/model_info.txt', 'w') as f:
        f.write(f'{session["user_id"]}_{uuid.uuid4()}_{acc}')
    
    return jsonify(success=True, accuracy=acc)

# db bounding
db_config = {
    'host': '140.138.155.243',         # e.g., 'localhost' or your remote server IP
    'user': 'CS380B',     # MySQL username
    'password': 'YZUCS380B', # MySQL password
    'database': 'CS380B'  # Database name
}

def add_user(username, password):
    """新增用戶到資料庫"""
    try:
        connection = pymysql.connect(**db_config)
        cursor = connection.cursor()

        # 插入用戶資料
        sql = "INSERT INTO users_cv (username, password) VALUES (%s, %s)"
        cursor.execute(sql, (username, password))
        connection.commit()
        return True
    except Exception as e:
        print(f"Error adding user: {e}")
        return False
    finally:
        if connection:
            connection.close()

def verify_user(username, password):
    """驗證用戶是否存在於資料庫"""
    try:
        # 連接到資料庫
        connection = pymysql.connect(**db_config)
        cursor = connection.cursor()

        # 查詢用戶
        sql = "SELECT * FROM users_cv WHERE username = %s AND password = %s"
        cursor.execute(sql, (username, password))
        user = cursor.fetchone()  

        return user is not None 
    finally:
        if connection:
            connection.close()


@app.route('/', methods=['GET', 'POST'])
def login_or_register():
    if request.method == 'POST':
        action = request.form.get('action')  
        username = request.form.get('user')
        password = request.form.get('password')

        if action == 'signin':  
            if verify_user(username, password):
                session['user_id'] = username
                return redirect(url_for('pose')) 
            else:
                return render_template('index.html', 
                                   alert_message='Account not found. Plz create an account',
                                   username=username, password=password)
        elif action == 'regist':  
            if add_user(username, password): 
                session['user_id'] = username
                return redirect(url_for('pose'))  
            else:
                return render_template('index.html', error='Registration failed. Please try again.')

    return render_template('index.html')

### pose.html 主頁
@app.route('/pose', methods=['POST', 'GET'])    
def pose():
    if 'user_id' not in session:
        return redirect(url_for('login'))  # 若未登入，跳轉回登入頁面
    return render_template('pose.html')





# add
@app.route('/visualize', methods=['GET'])
def visualize():
    connection = pymysql.connect(
        host=db_config['host'],
        user=db_config['user'],
        password=db_config['password'],
        database=db_config['database']
    )
    
    user_id = session['user_id']
    
    try:
        # 從資料庫中抓取數據
        query = "SELECT taskid, json_data FROM users_cv_data WHERE username = %s"
        cases = pd.read_sql(query, connection, params=(user_id,))
        
        if cases.empty:
            return jsonify({'error': 'No data available for visualization'})

        charts = []  # 存放所有圖表的 HTML
        for index, row in cases.iterrows():
            case_id = row['taskid']
            json_data = json.loads(row['json_data'])
            df = pd.DataFrame(json_data)
            df_resample = resample_data(df)

            # --- pie chart ---
            class_counts = df['classification'].value_counts()
            pie_fig = go.Figure()
            pie_fig.add_trace(
                go.Pie(
                    labels=class_counts.index,
                    values=class_counts.values,
                    hole=0.3,
                    opacity=0.8,
                    textinfo='percent+label',
                    marker=dict(colors=["red", "green"]),
                )
            )

            pie_fig.update_traces(
                textinfo='percent+label',
                textfont=dict(size=14),
                marker=dict(colors=["#FF6347", "#32CD32"]),
            )
            pie_fig.update_layout(
                height=300,
                width=300,
                margin=dict(t=10, b=10, l=10, r=10),  # 縮小外邊距
            )

            pie_html = pie_fig.to_html(full_html=True)

            # --- Line chart ---
            smoothed_scores = gaussian_filter1d(df_resample['prediction'], sigma=3)
            line_fig = FigureResampler(go.Figure())

            merged_intervals = merge_classification_intervals(df_resample)
            adjusted_intervals = adjust_short_intervals(merged_intervals, min_duration=timedelta(seconds=3))

            total_good_duration = sum(
                (end - start).total_seconds()
                for start, end, classification in merged_intervals
                if classification == "Good"
            )
            total_bad_duration = sum(
                (end - start).total_seconds()
                for start, end, classification in merged_intervals
                if classification == "Bad"
            )

            line_fig.add_trace(
                go.Scattergl(
                    name=(
                        f"Dynamic Posture Analysis (Good: {total_good_duration:.0f}s, "
                        f"Bad: {total_bad_duration:.0f}s)"
                    ),
                    mode="lines",
                    line=dict(color="blue", width=2),
                ),
                hf_x=df_resample['datetime'],
                hf_y=smoothed_scores,
            )

            for start_time, end_time, classification in adjusted_intervals:
                line_fig.add_vrect(
                        x0=start_time,
                        x1=end_time,
                        fillcolor="#32CD32" if classification == "Good" else "#FF6347",
                        opacity=0.1,
                        layer="below",
                        line_width=0,
                )

            line_fig.update_layout(
                # title=dict(
                #     text=(
                #         f"<b>Dynamic Posture Analysis</b><br>"
                #         f"Good: <span style='color:green;'>{total_good_duration:.0f}s</span>, "
                #         f"Bad: <span style='color:red;'>{total_bad_duration:.0f}s</span>"
                #     ),
                # x=0.5,  # 標題居中
                # font=dict(size=16),  # 增大字體
                # ),
                xaxis_title="Time",
                yaxis_title="Prediction Score",
                legend_title="Posture Type",
            )

            line_html = line_fig.to_html(full_html=False)

            charts.append({
                'case_id': case_id,
                'line_html': line_html,
                'pie_html': pie_html,
                'total_good_duration': total_good_duration,
                'total_bad_duration': total_bad_duration
            })

    except Exception as e:
        return jsonify({'error': str(e)})
    finally:
        connection.close()

    return render_template('visualize.html', charts=charts)

# add
def adjust_short_intervals(intervals, min_duration=timedelta(seconds=3)):

    """
    將短於指定閾值的區間標記成與相鄰長區間相同的顏色。
    :param intervals: 合併後的區域列表，每個元素是 (start_time, end_time, classification)
    :param min_duration: 長區間的最小時間跨度
    :return: 調整後的區域列表
    """
    adjusted_intervals = []

    for i, (start_time, end_time, classification) in enumerate(intervals):
        duration = end_time - start_time

        # 如果區間長於閾值，直接添加
        if duration >= min_duration:
            adjusted_intervals.append((start_time, end_time, classification))
        else:
            # 處理短區間，嘗試繼承相鄰長區間的顏色
            if i > 0:  # 如果有前一個區間
                prev_classification = adjusted_intervals[-1][2]
                adjusted_intervals.append((start_time, end_time, prev_classification))
            elif i < len(intervals) - 1:  # 如果有下一個區間
                next_classification = intervals[i + 1][2]
                adjusted_intervals.append((start_time, end_time, next_classification))
            else:
                # 如果是唯一的區間（無前無後），保持原分類
                adjusted_intervals.append((start_time, end_time, classification))

    return adjusted_intervals

# add
def merge_classification_intervals(df):
    intervals = []
    current_classification = None
    start_time = None

    for idx, row in df.iterrows():
        if current_classification is None: 
            current_classification = row['classification']
            start_time = row['datetime']
        elif row['classification'] != current_classification:
            intervals.append((start_time, row['datetime'], current_classification))
            current_classification = row['classification']
            start_time = row['datetime']

    intervals.append((start_time, df.iloc[-1]['datetime'], current_classification))
    return intervals

# add, resample data to 1-minute intervals
def resample_data(df):

    df['datetime'] = pd.to_datetime(df['datetime'])
    df['prediction'] = df['prediction'].astype(float)
    df = df.set_index('datetime')

    df_numeric = df['prediction'].resample('1s').mean()
    df_non_numeric = df['classification'].resample('1s').apply(lambda x: x.mode()[0] if not x.mode().empty else 'Unknown')
    df_resampled = pd.concat([df_numeric, df_non_numeric], axis=1).reset_index()

    return df_resampled

# 有bug
# local variable 'checked' referenced before assignment
# 這邊的問題是 checked 沒有被定義
# local variable 'final_json' referenced before assignment
# 這邊的問題是 final_json 沒有被定義
def save_classification_result(user_id, prediction, classification):
    # 構造數據
    timestamp = datetime.now().isoformat()
    new_result = {
        "datetime": timestamp,
        "prediction": round(prediction, 2),
        "classification": classification
    }

    connection = pymysql.connect(
        host=db_config['host'],
        user=db_config['user'],
        password=db_config['password'],
        database=db_config['database']
    )

    try:
        with connection.cursor() as cursor:
            checked = False
            final_json = None

            # search user's latest taskid and datetime
            query_fetch = """
            SELECT json_data FROM users_cv_data 
            WHERE username = %s
            """
            cursor.execute(query_fetch, (user_id,))
            result = cursor.fetchall()

            if not result:
                taskid = "1"  # 初始 taskid 為 1
                query_insert = """
                INSERT INTO users_cv_data (taskid, username, json_data)
                VALUES (%s, %s, %s)
                """
                cursor.execute(query_insert, (taskid, user_id, json.dumps([new_result])))
                connection.commit()
                print(f"New record created for user {user_id}, task {taskid}: {[new_result]}")
                return 


            # 確認是否需要分配新 taskid
            print("result.length:", len(result))
            length = len(result)
            if length > 0:
                checked = True
                last_taskid = length
                json_data = json.loads(result[length-1][0])
                # print("json_data:", json_data)
                totCnt = len(json_data)
                # print("totCnt:", totCnt)
                last_data = json_data[totCnt-1]

                if isinstance(last_data, list):
                    for item in last_data:
                        last_data = item
                        if isinstance(last_data, dict):
                            break

                # print("last_data:", last_data)
                last_datetime = datetime.fromisoformat(last_data['datetime'].strip('"'))  # 移除 JSON 格式的引號
                # print("last_datetime:", last_datetime)
                time_diff = (datetime.now() - last_datetime).total_seconds()
                # print("time_diff:", time_diff)

                if time_diff > 60:  # 時間間隔大於 1 分鐘
                    taskid = str(int(last_taskid)+1)  # 分配新 taskid
                else:
                    taskid = last_taskid  # 使用現有 taskid
                    # print("json_data!:", json_data)
                    json_data.append(new_result)  # 直接添加新數據
                    final_json = json.dumps(json_data)
                    # print("final_json:", final_json)
                    checked = False
            else:
                taskid = "1"  # 無任何記錄時初始化新 taskid

            
            if checked:
                query_update = """
                INSERT INTO users_cv_data (taskid, username, json_data)
                VALUES (%s, %s, %s)
                ON DUPLICATE KEY UPDATE json_data = %s
                """

                cursor.execute(query_update, (taskid, user_id, json.dumps([new_result]), json.dumps([new_result])))
                connection.commit()
                # print(f"Result saved for user {user_id}, task {taskid}: {[new_result]}")
            elif final_json:
                query_update = """
                UPDATE users_cv_data 
                SET json_data = %s
                WHERE taskid = %s AND username = %s
                """                
                cursor.execute(query_update, (final_json, taskid, user_id))
                connection.commit()
                # print(f"Result append for user {user_id}, task {taskid}: {final_json}")
    except Exception as e:
        print(f"Error saving classification result: {str(e)}")
    finally:
        connection.close()


if __name__ == '__main__':
    app.run(debug=True, port=5000)