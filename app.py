import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import tempfile
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, MobileNetV2
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
import urllib.request
from collections import Counter
import nltk
import heapq
import re
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')

url = 'https://github.com/tadiwamark/CaptionCraft/releases/download/v2.0/image_captioning_model.h5'
filename = url.split('/')[-1]

urllib.request.urlretrieve(url, filename)
# Load your trained model
model = tf.keras.models.load_model(filename)  

# Load Tokenizer
with open('tokenizer.pickle', 'rb') as handle:  # Update with actual path if needed
    tokenizer = pickle.load(handle)

max_length = 49

# Load MobileNetV2 Model for feature extraction
def load_mobilenet_model():
    base_model = MobileNetV2(input_shape=(224,224,3), include_top=False, pooling='avg', weights='imagenet')
    x = base_model.output
    x = Dense(4096, activation='relu')(x)
    return Model(inputs=base_model.input, outputs=x)

# Function to perform summarization
def summarize(text, num_of_sentences=5):
    sentence_list = nltk.sent_tokenize(text)
    stopWords = set(stopwords.words("english"))
    word_frequencies = {}
    for word in nltk.word_tokenize(text):
        if word.lower() not in stopWords:
            if word in word_frequencies.keys():
                word_frequencies[word] += 1
            else:
                word_frequencies[word] = 1
    
    maximum_frequncy = max(word_frequencies.values())
    for word in word_frequencies.keys():
        word_frequencies[word] = (word_frequencies[word]/maximum_frequncy)
    
    sentence_scores = {}
    for sent in sentence_list:
        for word, freq in word_frequencies.items():
            if word in sent.lower():
                if sent in sentence_scores.keys():
                    sentence_scores[sent] += freq
                else:
                    sentence_scores[sent] = freq
    
    summary_sentences = heapq.nlargest(num_of_sentences, sentence_scores, key=sentence_scores.get)
    return ' '.join(summary_sentences)


def int_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def generate_description(model, tokenizer, photo):
    in_text = 'startseq'
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length, padding='post')
        yhat = model.predict([photo, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = int_to_word(yhat, tokenizer)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'endseq':
            break
    in_text = in_text.replace('startseq', '').replace('endseq', '').strip()
    return in_text

def app():
    st.title('Video Description Generator')
    uploaded_file = st.file_uploader("Upload a video (max 2MB)", type=['mp4', 'avi', 'mov'])
    
    if uploaded_file is not None:
        st.write("File uploaded successfully!")
        for _ in range(1):
            st.markdown("<span style='color:blue'>We are now going to generate a frame by frame description then give a final description at the end.</span>", unsafe_allow_html=True)
            time.sleep(0.5)
        descriptions = []  # List to hold the descriptions of each frame
        try:
            tfile = tempfile.NamedTemporaryFile(delete=False) 
            tfile.write(uploaded_file.read())
            
            mobilenet_model = load_mobilenet_model()
            cap = cv2.VideoCapture(tfile.name)
            frameRate = cap.get(5)
            
            frame_descriptions = []
            
            while cap.isOpened():
                frameId = cap.get(1)
                ret, frame = cap.read()
            
                if not ret:
                    break
                if frameId % ((int(frameRate) + 1) * 1) == 0:
                    frame = cv2.resize(frame, (224, 224))
                    img = preprocess_input(np.expand_dims(frame, axis=0))
                    feature = mobilenet_model.predict(img)
                    st.write("Feature extracted using MobileNet model")
                    description = generate_description(model, tokenizer, feature)
                    descriptions.append(description)  # Append the description to the list
                    
                    
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    st.image(frame, caption='Processed Frame', use_column_width=True)
                    st.write(description)
                    
            final_text = ' '.join(descriptions)  # Join all the descriptions into a single string
            final_summary = summarize(final_text)  # Summarize the final text
            st.subheader('Final Video Summary')
            st.write(final_summary)
            
        except Exception as e:
            st.write(f"An error occurred: {str(e)}")
            
        st.write("Video processing completed")
        cap.release()

if __name__ == '__main__':
    app()
