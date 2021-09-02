from typing import Text
from flask import Flask, render_template, request, redirect, Response, url_for
from flask.helpers import url_for
import model,test
from  google_cloud.texttospeech import quickstart
from werkzeug.utils import secure_filename
import pandas as pd
import os 



import cv2
import numpy as np
import matplotlib.pyplot as plt


app = Flask(__name__)
app.debug = True


@app.route('/')
def index():
    return render_template('/index.html')



# file을 submit하는 페이지 
# /upload 의 페이지로 들어와서, upload.html의 파일을 렌더링하여 보여줌 
# # 여기서, upload.html은 프로젝트 폴더 내의 templates 폴더에 존재해야 함(default)

@app.route('/upload')
def render_file():
    return render_template('upload.html')

@app.route('/file_uploaded')
def file_uploaded():
    return render_template('file_uploaded.html')



# file이 submit되면 전달되는 페이지
# upload.html에서 form이 제출되면 /file_uploaded로 옮겨지게 되어 있음.
@app.route('/file_uploaded', methods = ['GET', 'POST'])
def upload_file():
    if request.method == 'POST': # POST 방식으로 전달된 경우
        f = request.files['file1']
        # 파일 객체 혹은 파일 스트림을 가져오고, html 파일에서 넘겨지는 값의 이름을 file1으로 했기 때문에 file1임. 
        f.save(f'static/img/{secure_filename(f.filename)}') # 업로드된 파일을 특정 폴더에저장하고, 
        #df_to_html = pd.read_csv(f'uploads/{secure_filename(f.filename)}').to_html() # html로 변환하여 보여줌
        global image_path
        image_path = f'static/img/{secure_filename(f.filename)}'
         
        # return image_path
        return render_template("/file_uploaded.html")

# 인자값을 image_path 로 받아 test.py실행 
@app.route("/view")
def view():
    test.yolo(image_path)
    global list_label
    list_label = test.yolo(image_path)
    return render_template("/view.html")

@app.route("/tts")
def tts():
    quickstart.run_quickstart(list_label)
    return render_template("/tts.html")

@app.route("/tts")
def audioplay():

    return render_template("/tts.html")


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5500, debug = True)    
    # model.photo()
    # Text = model.photo() 

    # print("app.py출력확인 : {}.".format(Text))
    # quickstart.run_quickstart(Text)
    
