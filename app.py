from typing import Text
from flask import Flask, render_template, request, redirect
import model
from  google_cloud.texttospeech import quickstart

app = Flask(__name__)
app.debug = True


@app.route('/')
def index():
    return render_template('/index.html')


if __name__ == '__main__' :
    model.photo()
    Text = model.photo()
    print("app.py출력확인 : {}.".format(Text))
    quickstart.run_quickstart(Text)
    # app.run()