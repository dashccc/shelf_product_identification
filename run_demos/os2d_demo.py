from flask import Flask, render_template, Response, request
import cv2
import datetime, time
import os, sys
import numpy as np
from threading import Thread
import boto3
from bbox_match import *

# 45 SKU in total:
# cls_name=['AHyzhywqps', 'ksbtwsdqps', 'nfsqrxxjwsdqps', 'qqdqtwsdqps', 'sqbbbtwqps', 'tqdbtwsdqps', 'sqbbxgwqps', 'nfsqfxbtwsdqps', 'yqslkmjwsdqpsA', 'yqslbtwsdqpsA', 'sodazyqtqps', 'wxqpbtwqps', 'yqslbtwsdqps', 'xcgsbtwwtqps', 'xcjfptwwtqps', 'wxqpbxgwqps', 'HPhppnmfwqps', 'nfsqcjygwsdqps', 'ksnmwsdqps', 'tyAHnmwfjyl', 'xchyyzwwtqps', 'yqslkmjwsdqps', 'kkkltwyl', 'yqslrsjwsdqps', 'yqslsmzsdqps', 'ksllwsdqps', 'xchylzwwtqps', 'xcblbxgrsjwwtqps', 'yqdxywsdqps', 'AHbtwlcwqps', 'qtssmtwyl', 'lqdlzwsdqps', 'jqdjjwsdqps', 'wxqpmywqps', 'qpsbptwyl', 'tyAHpgcfjyl', 'qnsnmwyl', 'nfsqmjtwsdqps', 'yqslxhptwsdqps', 'yrnrwyl', 'HPhppsmtfwqps', 'sodathytqps', 'sodakmjxrzqps', 'yrsmtwyl', 'ssnmwsdqps']
cls_name={'29':'元气森林葡萄味', 
        '33':'喜茶苏打水雪梨味', 
        '31':'元气森林橘子味', 
        '45':'元气森林菠萝味', 
        '26':'喜茶苏打水葡萄味', 
        '44':'元气森林乳酸菌味'}
cls=[]
# for i in range(len(cls_name)):
#     cls.append(str(i))
for key in cls_name.keys():
    cls.append(key)
    
# rule=[{'23':1,'4':2,'40':2,'16':2,'42':2,'41':2,'10':2,'9':1,'8':1},
#      {'2':2,'39':2,'33':2,'13':2,'18':2,'28':2,'23':3},
#      {'17':2,'43':2,'11':2,'14':2,'18':2,'31':2,'24':3},
#      {'37':2,'30':2,'15':2,'26':2,'25':2,'32':2,'38':3},
#      {'7':2,'34':2,'35':2,'27':2,'1':2,'3':2,'12':3},
#      {'36':2,'22':1,'35':2,'20':2,'0':2,'5':2,'21':3}]
rule=[{'29':2, '33':1, '31':2},
    {'45':2, '26':1, '44':2}]

global capture,rec_frame, grey, switch, neg, face, rec, out, render, bboxs
capture=0
grey=0
neg=0
face=0
switch=1
rec=0
test=0
render=0

#make shots directory to save pics
try:
    os.mkdir('./shots')
except OSError as error:
    pass

#Load pretrained face detection model    
net = cv2.dnn.readNetFromCaffe('./saved_model/deploy.prototxt.txt', './saved_model/res10_300x300_ssd_iter_140000.caffemodel')

#instatiate flask app  
app = Flask(__name__, template_folder='./templates')
app.config['MAX_CONTENT_LENGTH'] = 20971520


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

def convert_str_to_array(res_str):
    res_list = []

    res_str2 = res_str[1:-1]
    list1 = res_str2.split("], ")

    # get rid of the brackets in strings:
    for i in range(len(list1)):
        list1[i] = list1[i][1:]
        print(list1[i])
    list1[-1] = list1[-1][:-1]

    for item in list1:
        res_list.append(list(map(float, item.split(", "))))

    print("res_list:", res_list)
    return res_list

def save_image(frame, res_str):
    global render, bboxs
    # return "../shots/shot_2021-07-13 231103.803140.png"
    w=frame.shape[1]
    h=frame.shape[0] 
    print(w, h)

    # bboxs = []

    # convert res_str to res list:
    bboxs = convert_str_to_array(res_str)
    
    bboxs_np=np.zeros((len(bboxs),6))
    # print("bboxs_np",bboxs_np)
 
    # for i in range(len(res)):
    # for idx,i in enumerate(res):
        # print("idx and i: ", idx,i)
        # line=list(map(float,i[1:-1].split(" ")))
        # print("line: ", line)
        # if len(res[i]) !=0:
        #     bboxs.append([int(res[i][0]), int(float(res[i][2]) * w - 0.5 * float(res[i][4]) * w),
        #                   int(float(res[i][3]) * h - 0.5 * float(res[i][5]) * h),
        #                   int(float(res[i][2]) * w + 0.5 * float(res[i][4]) * w),
        #                   int(float(res[i][3]) * h + 0.5 * float(res[i][5]) * h)])
        #     bboxs_np[i]=np.array([int(float(res[i][2]) * w - 0.5 * float(res[i][4]) * w),
        #                   int(float(res[i][3]) * h - 0.5 * float(res[i][5]) * h),
        #                   int(float(res[i][2]) * w + 0.5 * float(res[i][4]) * w),
        #                   int(float(res[i][3]) * h + 0.5 * float(res[i][5]) * h),1,int(res[i][0])])
    
    for i in range(len(bboxs)):
        if len(bboxs[i]) !=0:
            bboxs_np[i]=np.array([int(bboxs[i][2]),
                          int(bboxs[i][3]),
                          int(bboxs[i][4]),
                          int(bboxs[i][5]),1,int(bboxs[i][0])])
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
    
    # for i in bboxs:
    #     color = (0,128,255)
    #     print(i)
    #     image = cv2.rectangle(frame, (int(i[2]),int(i[3])), (int(i[4]),int(i[5])), color, thickness)
    #     image = cv2.putText(image, str(int(i[0])), (int(i[2]), int(i[3])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, thickness)

    cv2.imwrite("static/result_images/output.jpg", image)

def is_out_of_stock(product_code, product_number):
    # stock_requirement_dict = {
    #     0: 3,
    #     1: 3,
    #     2: 3,
    #     3: 3,
    #     4: 3,
    #     5: 3,
    #     6: 3,
    #     7: 3,
    #     8: 3
    # }
    # if product_number < stock_requirement_dict[product_code]:
    if product_number < 1:
        # out of stock
        return True
    else:
        return False

def calculate_stock():
    global bboxs
    line_list = []
    
    # create product name dict from cls_name list:
    # product_name_dict = {}
    # i=0
    # for key in cls_name:
    #     product_name_dict[i] = key
    #     i += 1

    # create in_stock_dict to calculate the number of each SKU
    in_stock_dict = {}
    for key, item in cls_name.items():
        in_stock_dict[item] = 0

    # try:
    if len(bboxs)>0:
        for item in bboxs:
            key = cls_name[str(int(item[0]))]
            in_stock_dict[key] = in_stock_dict[key] + 1 
        print(in_stock_dict)
        
        # for drawing stock status:
        draw_stock_list = []
        for sku_name, quantity in in_stock_dict.items():
            draw_stock_list.append([sku_name, quantity])

        line_list.append('<table class="table_style"')
        i = 0
        # while (i<=40):
        #     line_list.append('<tr>')
        #     for j in range(5):
        #         print(i+j)
        #         if is_out_of_stock(i+j, in_stock_dict[i+j]):
        #             # out of stock: character is red
        #             line_list.append('<td class="result_style_outOfStock">' + product_name_dict[i+j] + '的库存为：' + str(in_stock_dict[i+j]) + '（需补货）</td>')
        #         else:
        #             # in stock: character is white as normal
        #             line_list.append('<td class="result_style">' + product_name_dict[i+j] + '的库存为：' + str(in_stock_dict[i+j]) + '</td>')
        #     line_list.append('</tr>')
        #     i += 5
        while (i<=3):
            line_list.append('<tr>')
            for j in range(3):
                print(i+j)
                if is_out_of_stock(i+j, draw_stock_list[i+j][1]):
                    # out of stock: character is red
                    line_list.append('<td class="result_style_outOfStock">' + draw_stock_list[i+j][0] + '的库存为：' + str(draw_stock_list[i+j][1]) + '（需补货）</td>')
                else:
                    # in stock: character is white as normal
                    line_list.append('<td class="result_style">' + draw_stock_list[i+j][0] + '的库存为：' + str(draw_stock_list[i+j][1]) + '</td>')
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
                    EndpointName='os2dv1',
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
                with open("test_pic/test6.jpeg", "rb") as fp:
                    body = fp.read()

                runtime = boto3.client("sagemaker-runtime",region_name="us-east-2")
                tic = time.time()

                response = runtime.invoke_endpoint(
                    EndpointName='os2dv1',
                    Body=body,
                    ContentType='application/x-image',
                )
                
                response_body = response["Body"].read()

                toc = time.time()
                
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