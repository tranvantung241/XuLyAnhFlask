import os
# from django.shortcuts import redirect
from flask import Flask, request, render_template, send_from_directory, flash, url_for
from werkzeug.utils import secure_filename
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys   
from flask import render_template_string 
from wtforms import Form, TextField 
import datetime
import random
import requests
import re

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist
from keras.models import model_from_json
from keras import backend as K
from keras.models import load_model

import cv2 as cv

import string

__author__ = 'TranTung'
app = Flask(__name__)
APP_ROOT = os.path.dirname(os.path.abspath(__file__))

# =========================================DECLARE===========================================================
url_image = "static/image/imgThu.png"
result_predict_MNIST = None
model_MNIST = None
result_predict_selfie = None
model_selfie = None

# =========================================FUNCTION===========================================================

def randomString(stringLength=10):
    """Generate a random string of fixed length """
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(stringLength))
# =========================================ROUTE===========================================================
 
#route home -----------------------------------------------------------------------------------------------
@app.route("/")
def index(): 
    return render_template("home.html")

 
# handwriting
@app.route("/handwriting")
def handwriting():
    global url_image
    url_image = "static/image/imgThu.png"
    global result_predict_MNIST
    # url_image = "/static/image/so2.jpg"
    return render_template("handwriting.html", url_image = url_image, 
        result_predict_MNIST= result_predict_MNIST)

# selfiie
@app.route("/selfie")
def selfie():
    global url_image
    url_image = "static/image/imgThu.png"
    global result_predict_selfiie
    # url_image = "/static/image/so2.jpg"
    return render_template("selfie.html", url_image = url_image, 
        result_predict_selfie= result_predict_selfie)
 
 
#route upload -----------------------------------------------------------------------------------------------
@app.route("/upload_handing", methods=['POST', 'GET'])
def upload_handing(): 

    global url_image 
    global result_predict_MNIST
    image_url = request.form['image_url']


    if image_url != "":
        print(image_url)  

        target = os.path.join(APP_ROOT, 'static/image/')
        print(target)
        if not os.path.isdir(target):
            os.mkdir(target)

        name = randomString()
        r = requests.get(image_url, allow_redirects=True)
        if r.headers['content-type'] in ["image/jpeg", "image/png"]:
            content = r.content
            open(target + name+".jpg", 'wb').write(content)
        url_image =  target +name+".jpg";
        url_image = url_image.replace(APP_ROOT,'') 
        
        print(url_image)

    elif  image_url == "" :
        print(APP_ROOT)
        target = os.path.join(APP_ROOT, 'static/image/')
        print(target)
        if not os.path.isdir(target):
            os.mkdir(target)

        print(request.files.getlist("file"))
        for upload in request.files.getlist("file"):
            print(upload)
            print("{} is the file name".format(upload.filename))
            filename = upload.filename
            # This is to verify files are supported
            ext = os.path.splitext(filename)[1]
            if ext in [".jpg", ".png", ".csv"]: 
            # if (ext == ".jpg") or (ext == ".png") or (ext == ".csv"):
                print("File supported moving on...")
            else:
                render_template("Error.html", message="Files uploaded are not supported...")
            destination = "/".join([target, filename])
            print("Accept incoming file:", filename)
            print("Save it to:", destination)
            upload.save(destination)
            print(destination)

            url_image = destination.replace(APP_ROOT,'') 
       
    return render_template("handwriting.html", url_image = url_image,
        result_predict_MNIST =result_predict_MNIST)


@app.route("/upload_selfie", methods=['POST', 'GET'])
def upload_selfie(): 

    global url_image 
    global result_predict_selfie
    image_url = request.form['image_url']


    if image_url != "":
        print(image_url)  

        target = os.path.join(APP_ROOT, 'static/image/')
        print(target)
        if not os.path.isdir(target):
            os.mkdir(target)

        name = randomString()
        r = requests.get(image_url, allow_redirects=True)
        if r.headers['content-type'] in ["image/jpeg", "image/png"]:
            content = r.content
            open(target + name+".jpg", 'wb').write(content)
        url_image =  target +name+".jpg";
        url_image = url_image.replace(APP_ROOT,'') 
        
        print(url_image)

    elif  image_url == "" :
        print(APP_ROOT)
        target = os.path.join(APP_ROOT, 'static/image/')
        print(target)
        if not os.path.isdir(target):
            os.mkdir(target)

        print(request.files.getlist("file"))
        for upload in request.files.getlist("file"):
            print(upload)
            print("{} is the file name".format(upload.filename))
            filename = upload.filename
            # This is to verify files are supported
            ext = os.path.splitext(filename)[1]
            if ext in [".jpg", ".png", ".csv"]: 
            # if (ext == ".jpg") or (ext == ".png") or (ext == ".csv"):
                print("File supported moving on...")
            else:
                render_template("Error.html", message="Files uploaded are not supported...")
            destination = "/".join([target, filename])
            print("Accept incoming file:", filename)
            print("Save it to:", destination)
            upload.save(destination)
            print(destination)

            url_image = destination.replace(APP_ROOT,'') 
       
    return render_template("selfie.html", url_image = url_image,
        result_predict_selfie =result_predict_selfie)


 
#route predict_MNIST -----------------------------------------------------------------------------------------------
@app.route('/predict_MNIST', methods=['POST', 'GET'])
def predict_MNIST(): 
    
    global result_predict_MNIST
    global model_MNIST
    global url_image

    
    
    if(model_MNIST ==  None):
        K.clear_session()
        model_MNIST = load_model('static/model/model_hand_writing.h5')
        print("Load model thanh cong")
    else:
        pass

 
    img = cv.imread(APP_ROOT+url_image, 0)
    print(img)
    print(url_image)
    img = cv.resize(img,(28,28))   

    
    result_predict_MNIST = model_MNIST.predict(img.reshape(1,28,28,1))
    result_predict_MNIST_1 = np.argmax(result_predict_MNIST)
    print(result_predict_MNIST_1)
    result_predict_MNIST = None
    return render_template("handwriting.html", url_image = url_image,
     result_predict_MNIST = result_predict_MNIST_1)

#route predict_selfie -----------------------------------------------------------------------------------------------
@app.route('/predict_selfie', methods=['POST', 'GET'])
def predict_selfie(): 
    
    global result_predict_selfie
    global model_selfie
    global url_image
 

    if(model_selfie ==  None):
        K.clear_session() 
        model_selfie = load_model('static/model/model_selfie_4000_v2_0.84.h5')
        print("Load model thanh cong")
    else:
        pass

 
    # img = cv.imread(APP_ROOT+url_image, 0)
    # print(img)
    # print(url_image)
    # img = cv.resize(img,(28,28))   

    
    # result_predict_selfie = model_MNIST.predict(img.reshape(1,28,28,1))
    # result_predict_selfie_1 = np.argmax(result_predict_selfie)
    # print(result_predict_selfie_1)
    # result_predict_selfie = None

    # ////
    num_channel = 1

    test_image = cv.imread(APP_ROOT+url_image)
    test_image=cv.cvtColor(test_image, cv.COLOR_BGR2GRAY)
    test_image=cv.resize(test_image,(128,128))
    test_image = np.array(test_image)
    test_image = test_image.astype('float32')
    test_image /= 255
    # print (test_image.shape)
       
    if num_channel==1:
        if K.image_dim_ordering()=='th':
            test_image= np.expand_dims(test_image, axis=0)
            test_image= np.expand_dims(test_image, axis=0)
    #       print (test_image.shape)
        else:
            test_image= np.expand_dims(test_image, axis=3) 
            test_image= np.expand_dims(test_image, axis=0)
    #       print (test_image.shape)
            
    else:
        if K.image_dim_ordering()=='th':
            test_image=np.rollaxis(test_image,2,0)
            test_image= np.expand_dims(test_image, axis=0)
    #       print (test_image.shape)
        else:
            test_image= np.expand_dims(test_image, axis=0)
    #       print (test_image.shape)
            
    # Predicting the test image
    print("Định nghĩa các class:  noselfie  = 0; selfie = 1")
    print((model_selfie.predict(test_image)))
    result_predict_selfie_1 = ""

    if (model_selfie.predict_classes(test_image) == 0):
        result_predict_selfie_1 = "No-selfie"
    else:
        result_predict_selfie_1 = "Selfie"

    result_predict_selfie = None
    return render_template("selfie.html", url_image = url_image,
     result_predict_selfie = result_predict_selfie_1)


#MAIN -----------------------------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(host='localhost', port=8181, debug=True)
