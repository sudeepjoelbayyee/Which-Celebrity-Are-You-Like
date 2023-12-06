import streamlit as st
from PIL import Image
import os
import cv2
from mtcnn import MTCNN
import numpy as np
from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import bz2

detector = MTCNN()
model = VGGFace(model='resnet50',include_top=False,input_shape=(224,224,3),pooling='avg')
feature_list = pickle.load(bz2.open('embeddings.pkl','rb'))
filenames = pickle.load(bz2.open('filenames.pkl','rb'))

def save_uploaded_img(uploaded_img):
    try:
        with open(os.path.join('uploads',uploaded_img.name),'wb') as f:
            f.write(uploaded_img.getbuffer())
        return True
    except:
        return False

def extract_features(img_path,model,detector):
    img = cv2.imread(img_path)
    results = detector.detect_faces(img)

    x, y, width, height = results[0]['box']

    face = img[y:y + height, x:x + width]

    # extract its features
    image = Image.fromarray(face)
    image = image.resize((224, 224))
    face_array = np.asarray(image)
    face_array = face_array.astype('float32')
    expanded_img = np.expand_dims(face_array, axis=0)
    proprocessed_img = preprocess_input(expanded_img)
    result = model.predict(proprocessed_img).flatten()
    return result
def recommend(feature_list,features):
    similarity = []
    for i in range(len(feature_list)):
        similarity.append(cosine_similarity(features.reshape(1, -1), feature_list[i].reshape(1, -1))[0][0])

    index_pos = sorted(list(enumerate(similarity)), reverse=True, key=lambda x: x[1])[0][0]
    return index_pos

st.sidebar.title("Which Bollywood Celebrity Are You?")
uploaded_img = st.file_uploader('Choose an Image')

if uploaded_img is not None:
    # save the image in a directory
    if save_uploaded_img(uploaded_img):
        # load the img
        display_img = Image.open(uploaded_img)

        # extact the features
        features = extract_features(os.path.join('uploads',uploaded_img.name),model,detector)
        # recommend
        index_pos = recommend(feature_list,features)
        predicted_actor = " ".join(filenames[index_pos].split('//')[1].split('_'))

        col1,col2 = st.columns(2)
        with col1:
            st.subheader("Your Uploaded Image")
            st.image(display_img,width=250)

        with col2:
            st.subheader("Seems like {}".format(predicted_actor))
            st.image(filenames[index_pos],width=250)
