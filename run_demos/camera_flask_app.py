from flask import Flask, render_template, Response, request
import cv2
import datetime, time
import os, sys
import numpy as np
from threading import Thread
import boto3
from bbox_match import *
cls=['0','1','2','3','4','5','6','7','8']
rule=[{'1':5,'2':6},
     {'3':6,'4':3,'5':6},
     {'6':6,'7':2,'8':6}]

global capture,rec_frame, grey, switch, neg, face, rec, out, render, bboxs
capture=0
grey=0
neg=0
face=0
switch=1
rec=0
test=0
render=0

# product reference
product_dict = {
    0: "欣和味达美味极鲜酱油1.8L",
    1: "欣和六月香甜面酱300g",
    2: "欣和六月香豆瓣酱300g",
    3: "欣和味达美味极鲜酱油800ml",
    4: "欣和味达美醇香米醋190ml",
    5: "欣和味达美味极鲜酱油1L",
    6: "欣和特级酱油500ml",
    7: "欣和味达美冰糖老抽酱油1L",
    8: "欣和葱伴侣黄豆酱900g",
}

#make shots directory to save pics
try:
    os.mkdir('./shots')
except OSError as error:
    pass

#Load pretrained face detection model    
net = cv2.dnn.readNetFromCaffe('./saved_model/deploy.prototxt.txt', './saved_model/res10_300x300_ssd_iter_140000.caffemodel')

#instatiate flask app  
app = Flask(__name__, template_folder='./templates')


camera = cv2.VideoCapture(0)
boto3.setup_default_session(profile_name='haoran')

def record(out):
    global rec_frame
    while(rec):
        time.sleep(0.05)
        out.write(rec_frame)


def detect_face(frame):
    global net
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
        (300, 300), (104.0, 177.0, 123.0))   
    net.setInput(blob)
    detections = net.forward()
    confidence = detections[0, 0, 0, 2]

    if confidence < 0.5:            
            return frame           

    box = detections[0, 0, 0, 3:7] * np.array([w, h, w, h])
    (startX, startY, endX, endY) = box.astype("int")
    try:
        frame=frame[startY:endY, startX:endX]
        (h, w) = frame.shape[:2]
        r = 480 / float(h)
        dim = ( int(w * r), 480)
        frame=cv2.resize(frame,dim)
    except Exception as e:
        pass
    return frame

def check_position(range_dict, item):
    if (item[0]>=range_dict['x11'] and item[0]<=range_dict['x12']) or (item[2]>=range_dict['x21'] and item[2]<=range_dict['x22']) or (item[1]>=range_dict['y11'] and item[1]<=range_dict['y12']) or (item[3]>=range_dict['y21'] and item[3]<=range_dict['y22']):
        return True
    return False

def save_image(frame, res_str):
    global render, bboxs
    # return "../shots/shot_2021-07-13 231103.803140.png"
    w=frame.shape[1]
    h=frame.shape[0] 
    print(w, h)

    bboxs = []

    # convert res_str to res list:
    res_str2 = res_str[1:-1]
    res = res_str2.split(", ")
    bboxs_np=np.zeros((len(res),6))
    for idx,i in enumerate(res):
        line=list(map(float,i[1:-1].split(" ")))
        if len(line) !=0:
            bboxs.append([int(line[0]), int(float(line[1]) * w - 0.5 * float(line[3]) * w),
                          int(float(line[2]) * h - 0.5 * float(line[4]) * h),
                          int(float(line[1]) * w + 0.5 * float(line[3]) * w),
                          int(float(line[2]) * h + 0.5 * float(line[4]) * h)])
            bboxs_np[idx]=np.array([int(float(line[1]) * w - 0.5 * float(line[3]) * w),
                          int(float(line[2]) * h - 0.5 * float(line[4]) * h),
                          int(float(line[1]) * w + 0.5 * float(line[3]) * w),
                          int(float(line[2]) * h + 0.5 * float(line[4]) * h),1,int(line[0])])
    
    print(len(bboxs))
    print(len(res))      
    print(bboxs_np)
    # draw rectangle
    product_location_range_dict = {
        0: {'x11':1051, 'x12':1200, 'y11':718, 'y12':900, 'x21':1091, 'x22':1291, 'y21':812, 'y22':1012},
        1: {'x11':1133, 'x12':1333, 'y11':634, 'y12':834, 'x21':1161, 'x22':1361, 'y21':658, 'y22':858},
        2: {'x11':1178, 'x12':1378, 'y11':521, 'y12':721, 'x21':1209, 'x22':1409, 'y21':545, 'y22':745},
        3: {'x11':1051, 'x12':1251, 'y11':573, 'y12':773, 'x21':1074, 'x22':1274, 'y21':662, 'y22':862},
        4: {'x11':1,'x12':1, 'y11':1, 'y12':1, 'x21':1, 'x22':1, 'y21':1, 'y22':1},
        5: {'x11':967, 'x12':1167, 'y11':480, 'y12':680, 'x21':992, 'x22':1192, 'y21':559, 'y22': 759},
        6: {'x11':1159, 'x12':1359, 'y11':577, 'y12':777, 'x21':1183, 'x22':1383, 'y21':656, 'y22':856},
        7: {'x11':1,'x12':1, 'y11':1, 'y12':1, 'x21':1, 'x22':1, 'y21':1, 'y22':1},
        8: {'x11':1272, 'x12':1472, 'y11':745, 'y12':945, 'x21':1304, 'x22':1504, 'y21':809, 'y22':1009}
    }
    thickness = 2

    idx=bboxs_np[:,-1]!=1
    bboxs_np=bboxs_np[idx]

    flag,true_bboxes,error_bboxes=relative_matching(cls, rule, bboxs_np)
    print(flag)
    for i in true_bboxes:
        color = (0,128,255)
        print(i)
        image = cv2.rectangle(frame, (int(i[0]),int(i[1])), (int(i[2]),int(i[3])), color, thickness)
        image = cv2.putText(image, str(int(i[5])), (int(i[0]), int(i[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, thickness)
    for i in error_bboxes:
        color = (42,0,202)
        image = cv2.rectangle(frame, (int(i[0]),int(i[1])), (int(i[2]),int(i[3])), color, thickness)
        image = cv2.putText(image, str(int(i[5])), (int(i[0]), int(i[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, thickness)
    #     else:
    # for i in bboxs:
    #     if check_position(product_location_range_dict[i[0]], [i[1],i[2],i[3],i[4]]):
    #         # good place
    #         color = (0,128,255) # color in BGR, orange
    #         image = cv2.rectangle(frame, (i[1],i[2]), (i[3],i[4]), color, thickness)
    #         image = cv2.putText(image, str(i[0]), (i[1], i[2]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, thickness)
    #     else:
    #         # wrong place
    #         color = (42,0,202) # color in BGR, red
    #         image = cv2.rectangle(frame, (i[1],i[2]), (i[3],i[4]), color, thickness)
    #         image = cv2.putText(image, str(i[0]), (i[1], i[2]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, thickness)

    cv2.imwrite("static/result_images/output.jpg", image)

def is_out_of_stock(product_code, product_number):
    stock_requirement_dict = {
        0: 3,
        1: 3,
        2: 3,
        3: 3,
        4: 3,
        5: 3,
        6: 3,
        7: 3,
        8: 3
    }
    if product_number < stock_requirement_dict[product_code]:
        # out of stock
        return True
    else:
        return False

def calculate_stock():
    global bboxs
    line_list = []
    product_name_dict = {
        0: "欣和味达美味极鲜酱油1.8L",
        1: "欣和六月香甜面酱300g",
        2: "欣和六月香豆瓣酱300g",
        3: "欣和味达美味极鲜酱油800ml",
        4: "欣和味达美醇香米醋190ml",
        5: "欣和味达美味极鲜酱油1L",
        6: "欣和特级酱油500ml",
        7: "欣和味达美冰糖老抽酱油1L",
        8: "欣和葱伴侣黄豆酱900g"
    }
    # try:
    if len(bboxs)>0:
        in_stock_dict = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0}
        for item in bboxs:
            in_stock_dict[item[0]] = in_stock_dict[item[0]] + 1 
        line_list.append('<table class="table_style"')
        i = 0
        while (i<=6):
            line_list.append('<tr>')
            for j in range(3):
                print(i+j)
                if is_out_of_stock(i+j, in_stock_dict[i+j]):
                    # out of stock: character is red
                    line_list.append('<td class="result_style_outOfStock">' + product_name_dict[i+j] + '的库存为：' + str(in_stock_dict[i+j]) + '（需补货）</td>')
                else:
                    # in stock: character is white as normal
                    line_list.append('<td class="result_style">' + product_name_dict[i+j] + '的库存为：' + str(in_stock_dict[i+j]) + '</td>')
            line_list.append('</tr>')
            i += 3
    else:
        pass
        # write no stock info

    with open('static/result_images/stock_results.txt', 'w') as f:
        f.writelines(line_list)
    # except:
    #     with open('static/result_images/results.txt', 'w') as f:
    #         f.write('<li class="result_style">请先点击按钮“拍照并分析”。</li>')

def gen_frames():  # generate frame by frame from camera
    global out, capture, rec_frame, test, render
    while True:
        success, frame = camera.read() 
        if success:
            if(face):                
                frame= detect_face(frame)
            if(grey):
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if(neg):
                frame=cv2.bitwise_not(frame)    
            if(capture):
                capture=0
                now = datetime.datetime.now()
                p = os.path.sep.join(['shots', "shot_{}.png".format(str(now).replace(":",''))])
                cv2.imwrite(p, frame)

                render=1

                body = cv2.imencode(".jpg", frame)[1].tobytes()
                runtime = boto3.client("sagemaker-runtime",region_name="us-east-2")
                response = runtime.invoke_endpoint(
                    EndpointName='yolov5v2',
                    Body=body,
                    ContentType='application/x-image',
                )
                body = response["Body"].read()
                res=body.decode()
                print(body.decode())

                save_image(frame, res)
                calculate_stock()

            if(test):
                test=0
                body = b""
                with open("test_pic/0_020.jpg", "rb") as fp:
                    body = fp.read()

                runtime = boto3.client("sagemaker-runtime",region_name="us-east-2")
                tic = time.time()

                response = runtime.invoke_endpoint(
                    EndpointName='yolov5v2',
                    Body=body,
                    ContentType='application/x-image',
                )
                
                response_body = response["Body"].read()

                toc = time.time()

                print(response_body.decode())
                print(f"elapsed: {(toc - tic) * 1000.0} ms")

                render = 1 
                nparr = np.fromstring(body, np.uint8)
                img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                save_image(img_np, response_body.decode())
                calculate_stock()
            
            if(rec):
                rec_frame=frame
                frame= cv2.putText(cv2.flip(frame,1),"Recording...", (0,25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),4)
                frame=cv2.flip(frame,1)
            
                
            try:
                ret, buffer = cv2.imencode('.jpg', cv2.flip(frame,1))
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            except Exception as e:
                pass
                
        else:
            pass


@app.route('/')
def index():
    return render_template('index.html')    
    
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/requests',methods=['POST','GET'])
def tasks():
    global switch,camera
    if request.method == 'POST':
        if request.form.get('click') == '拍照并分析':
            global capture
            capture=1
        elif  request.form.get('grey') == 'Grey':
            global grey
            grey=not grey
        elif  request.form.get('neg') == 'Negative':
            global neg
            neg=not neg
        elif  request.form.get('face') == 'Face Only':
            global face
            face=not face 
            if(face):
                time.sleep(4)   
        elif  request.form.get('stop') == '关闭摄像头/开启摄像头':
            
            if(switch==1):
                switch=0
                camera.release()
                cv2.destroyAllWindows()
                
            else:
                camera = cv2.VideoCapture(0)
                switch=1
        elif  request.form.get('rec') == 'Start/Stop Recording':
            global rec, out
            rec= not rec
            if(rec):
                now=datetime.datetime.now() 
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                out = cv2.VideoWriter('vid_{}.avi'.format(str(now).replace(":",'')), fourcc, 20.0, (640, 480))
                #Start new thread for recording the video
                thread = Thread(target = record, args=[out,])
                thread.start()
            elif(rec==False):
                out.release()
        elif request.form.get('test') == '测试':
            global test
            test=1
                 
    elif request.method=='GET':
        return render_template('index.html')
    return render_template('index.html')


if __name__ == '__main__':
    app.run()
    
camera.release()
cv2.destroyAllWindows()     