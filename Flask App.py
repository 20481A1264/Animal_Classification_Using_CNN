import numpy as np
import math
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask import Flask,render_template,request

app=Flask(__name__)

model=load_model("model.h5")

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict',methods=['GET','POST'])
def upload():
    if request.method=='POST':
        f=request.files['image']
        basepath=os.path.dirname(__file__)
        filepath=os.path.join(basepath,'uploads',f.filename)
        f.save(filepath)
        img=image.load_img(filepath,target_size=(64,64))
        pred=model.predict(img.reshape(1, 100, 100, 3))
        pred=pred>0.5
        index=["Dog","Cat"]
        text="The Classified Animal is : " +str(index[pred])
    return text
if __name__=='__main__':
    app.run(debug=False)