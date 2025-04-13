import streamlit as st
import cv2
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

model = load_model(r'D:\Work\Route\C07\S16\Project\covid_19_model.h5') # saved model = w , b --> freezed ----
class_map = {0: 'Covid',
             1: 'Normal',
             2: 'Viral Pneumonia'}                                                                              
##-------------Main FUNs--------------------##
def preprocessing(img,x_resize,y_resize,dim):
    #1- convert img --> array
    new_img = np.array(img)
    #2- resize the image
    new_img = cv2.resize(new_img,(x_resize,y_resize))
    #3- convert img --> grayscale
    if new_img.ndim ==2:
        new_img = cv2.cvtColor(new_img,cv2.COLOR_BGR2GRAY)
    #4- normalize
    new_img = new_img.astype('float32')/255.0
    #5- reshape the img --> to match the model input
    new_img= new_img.reshape(1,x_resize,y_resize,dim)
    return new_img

def prediction(img):
    pred = model.predict(img)
    pred_labels = np.argmax(pred, axis = 1)
    pred_class= class_map[pred_labels[0]]
    return pred_class
##----------------GUI-----------------##
st.title('Covid-19 Detection Application')
st.write('This is a simple application to detect Covid-19 from Chest X-ray images')
uploaded_img = st.file_uploader('Upload an image, plz: ',type = ['jpg','png','jpeg'])
if uploaded_img is not None:
    img = Image.open(uploaded_img)
    st.image(img, caption = 'uploaded_image!' )

    if st.button('Predict'):
        if img:
            new_img = preprocessing(img,224,224,3)
            pred_labels = prediction(new_img)
            st.write(f'This image represent: ({pred_labels}) class')
        else:
            st.write('Please, upload an image')
