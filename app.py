from typing import Text
from flask import Flask, render_template, request, redirect, Response, url_for
from flask.helpers import url_for
import test
from  google_cloud.texttospeech import quickstart
from werkzeug.utils import secure_filename
import pandas as pd
import random

app = Flask(__name__)
app.debug = True


@app.route('/')
def index():
    return render_template('/index.html')


@app.route('/upload')
def render_file():
    return render_template('upload.html')


@app.route('/spell')
def spell():
    classesFile = 'coco.names'
    classNames = []

    with open(classesFile,'rt') as f :
        classNames = f.read().rstrip('\n').split('\n')

    spell_result = random.sample(classNames,1)

    return render_template('spell.html',spell_result=spell_result)


@app.route('/about')
def about():
    return render_template('about.html')



@app.route('/file_uploaded')
def file_uploaded():
    return render_template('file_uploaded.html')



# file이 submit되면 전달되는 페이지
# upload.html에서 form이 제출되면 /file_uploaded로 옮겨짐
@app.route('/file_uploaded', methods = ['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['file1']
        # 파일 객체 혹은 파일 스트림을 가져오고, html 파일에서 넘겨지는 값의 이름을 file1으로 지정했기에. 
        f.save(f'static/img/{secure_filename(f.filename)}') 
        global image_path
        image_path = f'static/img/{secure_filename(f.filename)}'
        return render_template("/file_uploaded.html")


@app.route("/view")
def view():
    test.yolo(image_path)
    global list_label
    list_label = test.yolo(image_path)
    return render_template("/view.html", list_label=list_label)


@app.route("/tts")
def tts():
    quickstart.run_quickstart(list_label)
    return render_template("/tts.html")


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=80, debug = True)    
    
