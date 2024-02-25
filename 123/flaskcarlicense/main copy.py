# main.py
import base64
from collections import Counter
import cv2
from database import db
import datetime
from datetime import datetime, timedelta
from flask import Flask, render_template, request, redirect, url_for,jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_socketio import SocketIO, emit
from io import BytesIO
import json
from models import User
import numpy as np
import os
import requests
import threading
from ultralytics import YOLO
import qrcode

app = Flask(__name__)
project_dir = os.path.dirname(os.path.abspath(__file__))
db_path = os.path.join(project_dir, 'carlicense.db')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///{}'.format(db_path)
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)


def get_utc_plus_8_hours():
    return datetime.utcnow() + timedelta(hours=8)


class InRecord(db.Model):
    __tablename__ = 'in_record'
    id = db.Column(db.Integer, primary_key=True)
    license_plate = db.Column(db.String(64), nullable=False)
    image_data = db.Column(db.LargeBinary, nullable=False)  # 存储二进制图像数据
    entry_time = db.Column(db.DateTime, default=get_utc_plus_8_hours)  # 使用函數設置預設值
    exit_time = db.Column(db.DateTime, default=None, nullable=True)  # 新增離場時間欄位
    ecpay_url = db.Column(db.String, default=None, nullable=True)  # 新增訂單URL欄位
    linepay_url = db.Column(db.String, default=None, nullable=True)  # 新增訂單URL欄位
    def __repr__(self):
        return f'<InRecord {self.license_plate}>'
with app.app_context():
    db.create_all()



    
@app.route('/hello/<name>')
def hello(name):
    return render_template('page.html', name=name)
    
@app.route('/video')
def goto_camera():
    return render_template('html5_camera.html')


@app.route('/out_video')
def goto_out_camera():
    return render_template('out_camera.html')
    
    
@app.route('/item')
def goto_item():
    return render_template('list_item.html')
    
    
def get_current_user():
    return {'name':'joseph','password':'1234'}
    
@app.route("/me")
def me_api():
    user = get_current_user()
    print(user)
    return {
        "username": user['name'],
        "password": user['password']
    }

#-------------websocket------------------------    
socketio = SocketIO(app)
socketio.init_app(app, cors_allowed_origins="*")

# flaskcarlicense\flaskcarlicense\upload
def save_img(msg):

    filename=datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+'.png'
    base64_img_bytes = msg.encode('utf-8')
    with open('./flaskcarlicense/upload/'+filename, "wb") as save_file:
        save_file.write(base64.decodebytes(base64_img_bytes))


#user defined event 'client_event'
@socketio.on('client_event')
def client_msg(msg):
    #print('received from client:',msg['data'])
    emit('server_response', {'data': msg['data']}, broadcast=False) #include_self=False

#user defined event 'connect_event'
@socketio.on('connect_event')   
def connected_msg(msg):
    print('received connect_event')
    emit('server_response', {'data': msg['data']})
    
    
#user defined event 'capture_event'
@socketio.on('capture_event')   
def connected_msg(msg):
    print('received capture_event')
    #print(msg)
    save_img(msg)
    #here we just send back the original image to browser.
    #maybe, you can do image processinges before sending back 
    emit('capture_event', msg, broadcast=False)
    






latest_frame = None

def display_frames():
    global latest_frame
    while True:
        if latest_frame is not None:
            cv2.imshow('frame', latest_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

# 在一個新的執行緒中啟動 display_frames 函數
# thread = threading.Thread(target=display_frames)
# thread.start()

# @socketio.on('frame_event')   
# def handle_frame(msg):
#     global latest_frame
#     print('received frame')
#     frame_data = base64.b64decode(msg)
#     nparr = np.frombuffer(frame_data, np.uint8)
#     latest_frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
#     print('first0',latest_frame)

    
model_path = r'newTW.pt'

model = YOLO(model_path)

#line pay開始
def Linepay_generate_payment_url(fee_park):
    # Line Pay API 的測試環境基礎 URL
    base_url = 'https://sandbox-api-pay.line.me'

    # 設定 Line Pay 的測試用 Channel ID 和 Channel Secret
    channel_id = '2003576266'
    channel_secret = 'b251b8c728d92f31da4f2698d9e86634'

    # 設定 API 的 endpoint
    endpoint = '/v2/payments/request'

    # 設定 headers
    headers = {
        'Content-Type': 'application/json',
        'X-LINE-ChannelId': channel_id,
        'X-LINE-ChannelSecret': channel_secret,
    }

    payload = {
        'productName': '停車費',
        'amount': fee_park,
        'currency': 'TWD',
        'orderId': 'ORDER1234567890',
        'confirmUrl': 'http://127.0.0.1:5000/payment_callback',
        'cancelUrl': 'https://example.com/cancel',
    }

    # 發送 POST 請求到 Line Pay API
    response = requests.post(base_url + endpoint, headers=headers, data=json.dumps(payload))

    # 解析回傳結果
    if response.status_code == 200:
        linepay_url = response.json()
        payment_url = linepay_url.get('info', {}).get('paymentUrl', {}).get('web')
        if payment_url:
            return payment_url
        else:
            print("未找到 paymentUrl 或 web 鍵")
            return None
    else:
        print("發生錯誤，HTTP 狀態碼:", response.status_code)
        print("錯誤訊息:", response.text)
        return None

#Line pay結束

#綠界開始
import importlib.util
def Ecpay_generate_html_form(fee_park):
    spec = importlib.util.spec_from_file_location(
        "ecpay_payment_sdk",
        r"flaskcarlicense\ecpay_payment_sdk.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    order_params = {
        'MerchantTradeNo': datetime.now().strftime("NO%Y%m%d%H%M%S"),
        'StoreID': '',
        'MerchantTradeDate': datetime.now().strftime("%Y/%m/%d %H:%M:%S"),
        'PaymentType': 'aio',
        'TotalAmount': fee_park,
        'TradeDesc': '訂單測試',
        'ItemName': '停車費',
        'ReturnURL': 'http://127.0.0.1:5000/payment_callback',
        'ChoosePayment': 'ALL',
        'ClientBackURL': 'https://www.ecpay.com.tw/client_back_url.php',
        'ItemURL': 'https://www.ecpay.com.tw/item_url.php',
        'Remark': '交易備註',
        'ChooseSubPayment': '',
        'OrderResultURL': 'https://www.ecpay.com.tw/order_result_url.php',
        'NeedExtraPaidInfo': 'Y',
        'DeviceSource': '',
        'IgnorePayment': '',
        'PlatformID': '',
        'InvoiceMark': 'N',
        'CustomField1': '',
        'CustomField2': '',
        'CustomField3': '',
        'CustomField4': '',
        'EncryptType': 1,
    }

    extend_params_1 = {
        'ExpireDate': 7,
        'PaymentInfoURL': 'https://www.ecpay.com.tw/payment_info_url.php',
        'ClientRedirectURL': '',
    }

    extend_params_2 = {
        'StoreExpireDate': 15,
        'Desc_1': '',
        'Desc_2': '',
        'Desc_3': '',
        'Desc_4': '',
        'PaymentInfoURL': 'https://www.ecpay.com.tw/payment_info_url.php',
        'ClientRedirectURL': '',
    }

    inv_params = {}

    ecpay_payment_sdk = module.ECPayPaymentSdk(
        MerchantID='2000132',
        HashKey='5294y06JbISpM5x9',
        HashIV='v77hoKGq4kWxNNIS'
    )

    order_params.update(extend_params_1)
    order_params.update(extend_params_2)
    order_params.update(inv_params)

    try:
        final_order_params = ecpay_payment_sdk.create_order(order_params)
        action_url = 'https://payment-stage.ecpay.com.tw/Cashier/AioCheckOut/V5'  # 測試環境
        # action_url = 'https://payment.ecpay.com.tw/Cashier/AioCheckOut/V5' # 正式環境
        html = ecpay_payment_sdk.gen_html_post_form(action_url, final_order_params)
        return html
    except Exception as error:
        print('An exception happened: ' + str(error))
        return None
#綠界結束

def generate_qr_code_data(url):
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,
        border=4,
    )
    qr.add_data(url)
    qr.make(fit=True)

    img = qr.make_image(fill_color="black", back_color="white")

    # 將QR碼圖片轉換為BytesIO對象
    img_buffer = BytesIO()
    img.save(img_buffer, format='PNG')
    img_buffer.seek(0)
    
    # 讀取BytesIO對象的二進制數據
    img_data = img_buffer.getvalue()
    img_buffer.close()
    
    return img_data

def convert_to_base64(data):
    return base64.b64encode(data).decode('utf-8')

def insert_record(license_plate,encoded_image):

    existing_record = InRecord.query.filter(
        InRecord.license_plate == license_plate, 
        InRecord.exit_time.isnot(None),
        ).first()    
    encoded_image = base64.b64decode(encoded_image)

    if not existing_record:
        new_record = InRecord(license_plate=license_plate, image_data=encoded_image)
        db.session.add(new_record)
        db.session.commit()
        print("儲存成功")
    else:
        print("儲存失敗")

def insert_out_record(license_plate):
    # 尋找最近30分鐘內進入且車牌號碼匹配的記錄
    existing_record = InRecord.query.filter(InRecord.license_plate == license_plate, InRecord.exit_time.is_(None)).first()

    if existing_record:
        # 如果找到匹配的記錄，更新 exit_time 和 url
        existing_record.exit_time = datetime.utcnow() + timedelta(hours=8)
        existing_record.ecpay_url = 'aaa'  # 更新 ecpay_url 欄位
        existing_record.linepay_url = 'bbb'  # 更新 linepay_url 欄位 
        db.session.commit()  # 提交更改到數據庫
        print("記錄更新成功")
    else:
        # 如果沒有找到匹配的記錄，打印一條消息
        print("未找到匹配的進場記錄")




detected_license_list = []


@socketio.on('frame_event')   
def handle_frame(msg):
    global latest_frame, detected_license_list  # 引用全局变量
    print('received frame')

    # 解碼影像
    frame_data = base64.b64decode(msg)

    nparr = np.frombuffer(frame_data, np.uint8)

    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)


    try:
        results = model.predict(source=frame, conf=0.35)
    except Exception as e:
        print(f"An error occurred: {e}")

    result = results[0]    
    # print('result',result)

    # 假設車牌號碼的 class ID 是一個介於 0 到 36 之間的整數，分別對應於'-0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    plate_numbers = '-0123456789ABCDEFGHIJKLMNPQRSTUVWXYZ'

    detected_license = []
    detected_x_list = []

    average_index = 0
    average_sum = 0
    average = 0

    for box in result.boxes:
        bbox = box.xyxy.tolist()
        x1, y1, x2, y2 = int(bbox[0][0]), int(bbox[0][1]), int(bbox[0][2]), int(bbox[0][3])
        average_sum += (int(y2) - int(y1))
        average_index += 1
        # detected_yrange_list.append(int(y2) - int(y1))
    if average_index != 0:
        average = average_sum / average_index / 2

    for box in result.boxes:
        bbox = box.xyxy.tolist()
        x1, y1, x2, y2 = int(bbox[0][0]), int(bbox[0][1]), int(bbox[0][2]), int(bbox[0][3])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        y_range_detector = (int(y2) - int(y1))

        # 獲取車牌號碼的 class ID
        plate_class_id = int(box.cls)
        
        # 根據 class ID 獲取車牌號碼
        plate_number = plate_numbers[plate_class_id]
        
        # 在圖像上顯示車牌號碼
        cv2.putText(frame, f'{plate_number}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        print('Plate number:', plate_number)
        index = 0
        # if detected_x_list != []:
        #     for x in detected_x_list:
        #         if x < x1:
        #             detected_x_list.insert(index, x1)
        #             detected_license.insert(index, plate_number)
                    
        #             break
        #         index += 1
        # else:
        detected_x_list.append(x1)
        detected_license.append(plate_number)

        combined = sorted(zip(detected_x_list, detected_license), key=lambda x: x[0])

        # 使用 zip(*) 函數分解排序後的列表
        detected_x_list, detected_license = zip(*combined)

        # 將結果從元組轉換回列表
        detected_x_list = list(detected_x_list)
        detected_license = list(detected_license)

    detected_license_string = ''.join(detected_license)

    print('detected license', detected_license_string)

    detected_license_list.append(detected_license_string)
    license_count = Counter(detected_license_list)  # 使用 Counter 统计每个车牌号码出现的次数


    latest_frame = frame
    # print('66666')

        # 將處理後的圖像轉換為 base64 以便通過 WebSocket 發送
    _, buffer = cv2.imencode('.jpg', frame)
    encoded_image = base64.b64encode(buffer).decode('utf-8')


    # 發送到客戶端
    if license_count[detected_license_string] > 1 and len(detected_license_string) > 5:
        emit('update_image', {'image': encoded_image})
        emit('update_text', {'text': detected_license_string})
        insert_record(detected_license_string,encoded_image)


@socketio.on('out_frame_event')   
def handle_out_frame(msg):
    global latest_frame, detected_license_list  # 引用全局变量
    print('received frame')

    # 解碼影像
    frame_data = base64.b64decode(msg)

    nparr = np.frombuffer(frame_data, np.uint8)

    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)


    try:
        results = model.predict(source=frame, conf=0.35)
    except Exception as e:
        print(f"An error occurred: {e}")

    result = results[0]    
    # print('result',result)

    # 假設車牌號碼的 class ID 是一個介於 0 到 36 之間的整數，分別對應於'-0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    plate_numbers = '-0123456789ABCDEFGHIJKLMNPQRSTUVWXYZ'

    detected_license = []
    detected_x_list = []

    average_index = 0
    average_sum = 0
    average = 0

    for box in result.boxes:
        bbox = box.xyxy.tolist()
        x1, y1, x2, y2 = int(bbox[0][0]), int(bbox[0][1]), int(bbox[0][2]), int(bbox[0][3])
        average_sum += (int(y2) - int(y1))
        average_index += 1
        # detected_yrange_list.append(int(y2) - int(y1))
    if average_index != 0:
        average = average_sum / average_index / 2

    for box in result.boxes:
        bbox = box.xyxy.tolist()
        x1, y1, x2, y2 = int(bbox[0][0]), int(bbox[0][1]), int(bbox[0][2]), int(bbox[0][3])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        y_range_detector = (int(y2) - int(y1))

        # 獲取車牌號碼的 class ID
        plate_class_id = int(box.cls)
        
        # 根據 class ID 獲取車牌號碼
        plate_number = plate_numbers[plate_class_id]
        
        # 在圖像上顯示車牌號碼
        cv2.putText(frame, f'{plate_number}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        print('Plate number:', plate_number)
        index = 0
        # if detected_x_list != []:
        #     for x in detected_x_list:
        #         if x < x1:
        #             detected_x_list.insert(index, x1)
        #             detected_license.insert(index, plate_number)
                    
        #             break
        #         index += 1
        # else:
        detected_x_list.append(x1)
        detected_license.append(plate_number)

        combined = sorted(zip(detected_x_list, detected_license), key=lambda x: x[0])

        # 使用 zip(*) 函數分解排序後的列表
        detected_x_list, detected_license = zip(*combined)

        # 將結果從元組轉換回列表
        detected_x_list = list(detected_x_list)
        detected_license = list(detected_license)

    detected_license_string = ''.join(detected_license)

    print('detected license', detected_license_string)

    detected_license_list.append(detected_license_string)
    license_count = Counter(detected_license_list)  # 使用 Counter 统计每个车牌号码出现的次数


    latest_frame = frame
    # print('66666')

        # 將處理後的圖像轉換為 base64 以便通過 WebSocket 發送
    _, buffer = cv2.imencode('.jpg', frame)
    encoded_image = base64.b64encode(buffer).decode('utf-8')


    # 發送到客戶端
    if license_count[detected_license_string] > 1 and len(detected_license_string) > 5:
        existing_record = InRecord.query.filter_by(license_plate=detected_license_string).filter(InRecord.exit_time.is_(None)).first()
        if existing_record:
            # 從資料庫提取的二進制圖像數據
            image_data = existing_record.image_data

            # 將二進制數據轉換為NumPy數組
            nparr = np.frombuffer(image_data, np.uint8)

            # 將NumPy數組解碼成圖像
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            # 檢查圖像是否成功解碼
            if image is not None:
                # 使用 cv2.imencode 重新編碼圖像
                _, buffer = cv2.imencode('.jpg', image)
                # 如果需要，可以進一步處理 buffer 或將其儲存

                # 例如，將編碼後的圖像數據轉換為 base64 以便在網頁上顯示
                encoded_image2 = base64.b64encode(buffer).decode('utf-8')
            insert_out_record(detected_license_string)

            entry_time = existing_record.entry_time
            formatted_entry_time = entry_time.strftime('%Y-%m-%d %H:%M:%S')

            exit_time = existing_record.exit_time
            formatted_exit_time = exit_time.strftime('%Y-%m-%d %H:%M:%S')

            ecpay_url = existing_record.ecpay_url
            qr_code_ecpay = generate_qr_code_data(ecpay_url)
            base64_ecpay = convert_to_base64(qr_code_ecpay)
            linepay_url = existing_record.linepay_url
            qr_code_linepay = generate_qr_code_data(linepay_url)
            base64_linepay = convert_to_base64(qr_code_linepay)
            print('a111')
            timegap = exit_time - entry_time
            timegap_minutes = timegap.total_seconds() / 60
            global fee_func
            def fee_func(timegap_minutes):
                fee = (int(timegap_minutes // 30) + 1) * 25
                return min(fee, 190)
            payment = fee_func(timegap_minutes)
            global html_form
            html_form = Ecpay_generate_html_form(fee_func(timegap_minutes))
            ecpay_url = "http://127.0.0.1:5000/ecpay"
            linepay_url = Linepay_generate_payment_url(fee_func(timegap_minutes))
            print('a222')

            emit('qr_code_ecpay', {'image':base64_ecpay})
            emit('qr_code_linepay',{'image':base64_linepay})
            emit('update_out_image', {'image': encoded_image})
            emit('previous_image', {'image': encoded_image2})
            emit('update_out_text', {'text': detected_license_string})
            emit('have_license', {'text': ''})
            emit('formatted_entry_time', {'text': formatted_entry_time})
            emit('formatted_exit_time', {'text': formatted_exit_time})
            emit('ecpay_url', {'text': ecpay_url})
            emit('linepay_url', {'text': linepay_url})
            emit('payment',{'text':payment})
            emit('timegap', {'text': timegap_minutes})
            print('a333')

        else:
            emit('update_out_image', {'image': encoded_image})
            emit('update_out_text', {'text': detected_license_string})
            emit('no_license', {'text': '這個車牌號沒有入場紀錄'})


@app.route('/ecpay')
def ecpay_url():
    return render_template('ecpay_template.html', html_form=html_form)

    
#交易成功或失敗後回傳的route
@app.route('/payment_callback', methods=['POST'])
def payment_callback():
    # 解析接收到的付款通知數據
    payment_data = request.json  #付款通知是以 JSON 格式發送
    payment_status_greenpay = payment_data.get('TransMsg')      #綠界
    payment_status_linepay = payment_data.get('returnMessage') #linepay
    
    if payment_status_greenpay == 'Success' or payment_status_linepay == 'success':
        return render_template('success.html')
    else:
        return render_template('fail.html')



@app.route('/')
def index():
    users = User.query.all()
    return render_template('index.html', message="Welcome to your Flask app    🚅", users=users)

@app.route('/add_user', methods=['POST'])
def add_user():
    username = request.form['username']
    email = request.form['email']
    new_user = User(username=username, email=email)
    db.session.add(new_user)
    db.session.commit()
    return redirect(url_for('index'))

@app.route('/delete_user/<int:user_id>')
def delete_user(user_id):
    user_to_delete = User.query.get(user_id)
    if user_to_delete:
        db.session.delete(user_to_delete)
        db.session.commit()
    return redirect(url_for('index'))
if __name__ == '__main__':
    
    app.run(debug=True, port=os.getenv("PORT", default=5000))
    
