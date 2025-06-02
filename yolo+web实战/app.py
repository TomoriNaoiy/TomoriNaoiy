from flask import Flask,render_template,request,redirect,url_for
import os #os是用于路径凭借 创建目录等文件操作的库
from werkzeug.utils import secure_filename #一个将上传文件的文件名转化为安全格式的库
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from yolo_detect import detect_image #yolo模型检测
import cv2
from flask import Response
import torch
app=Flask(__name__)

UPLOAD_FOLDER='static/uploads' #将上传的图片保存在这个文件夹内
DETECTED_FOLDER='static/detected' #检测后的结果保存
app.config['UPLOAD_FOLDER']=UPLOAD_FOLDER
app.config['DETECTED_FOLDER']=DETECTED_FOLDER


os.makedirs(UPLOAD_FOLDER,exist_ok=True)# 如果用户没有 先创建一个该文件夹
os.makedirs(DETECTED_FOLDER,exist_ok=True)

def gen_frames(): # 摄像头调用 
    cap=cv2.VideoCapture(0) #打开本地摄像头（浏览器） 0时默认摄像头
    model=torch.hub.load("ultralytics/yolov5",'yolov5s',trust_repo=True)
    while True:
        success,frame=cap.read() #读取单独一帧
        if not success:
            break
        results=model(frame) #处理这一帧的图片 获得结果
        results.render()#渲染画布
        annotated_frame = results.ims[0]  # 获取渲染后的图像（RGB）
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)  # 转换为BGR再输出

        ret,buffer=cv2.imencode('.jpg',annotated_frame)#转化流
        frame_bytes=buffer.tobytes()
        yield(b'--frame\r\n'
              b'Content-Type: image/jpeg\r\n\r\n'+frame_bytes+b'\r\n') #以视频格式流出
        
@app.route("/",methods=['GET','POST'])
def index():
    if request.method=='POST':
        file=request.files['image'] #获得表单上传的images
        if file.filename=='':
            return "没有图片啊"
        filename=secure_filename(file.filename)
        file_path=os.path.join(app.config['UPLOAD_FOLDER'],filename)
        file.save(file_path) #保存完整路径
        detect_image(file_path,app.config['DETECTED_FOLDER'])
        return redirect(url_for('result',filename=filename))#跳转到该页面 展示图片``
    
    else:
        return render_template("index.html")
    
@app.route("/result/<filename>")#<filename>是url_for传来的参数
def result(filename):
    img_path=url_for('static',filename=f"detected/{filename}") #拼接相对路径
    return render_template("result.html",img_path=img_path)

@app.route("/video")
def video():
    return render_template("video.html")

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame') #这段表明也将多张照片做为视频流
    

        
if __name__=="__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))

