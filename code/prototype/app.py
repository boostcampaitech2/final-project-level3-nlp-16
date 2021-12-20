import streamlit as st
import time
import requests
import io
from PIL import Image
import os
import sys

from streamlit_lottie import st_lottie, st_lottie_spinner
from transformers import AutoTokenizer

from models.mmclf.mmclf import MultimodalCLF, get_model, predict_from_multimodal, get_config, get_tokenizer
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from text_extraction.text_extraction import extract_text
import pororo 

def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_url_hello = "https://assets5.lottiefiles.com/packages/lf20_V9t630.json"
lottie_url_download = "https://assets7.lottiefiles.com/packages/lf20_wsy1p3ad.json"
lottie_url_loading = "https://assets10.lottiefiles.com/packages/lf20_w6xlywkv.json"
lottie_hello = load_lottieurl(lottie_url_hello)
lottie_download = load_lottieurl(lottie_url_download)
lottie_loading = load_lottieurl(lottie_url_loading)

st.set_page_config()
title=''
content=''
st.title("당신도 중고왕이 될 수 있습니다!")


st.clf_model = get_model()
st.clf_tokenizer = get_tokenizer()
st.clf_config = get_config()

custom_bg_img = st.file_uploader(
    "상품 이미지를 올려주세요!", 
    type=["png", "jpg", "jpeg"]
)

if custom_bg_img:
    image_bytes = custom_bg_img.getvalue()
    image = Image.open(io.BytesIO(image_bytes))
    st.image(image, caption='Uploaded Image', width=300)

title=st.text_input("상품 제목을 입력해주세요.")

if custom_bg_img and title:
    with st_lottie_spinner(lottie_loading):
        labels = predict_from_multimodal(model=st.clf_model, tokenizer=st.clf_tokenizer, image_bytes=image_bytes, title=title, config=st.clf_config)
    st.write(labels)


content=st.text_area("내용을 입력해주세요.")

if st.button("해시태그 생성"):
    
    input_text = title + ' ' + content
    extraction_tag = extract_text(input_text)
    for tag in extraction_tag:
        st.write('#' + tag)
    
    with st_lottie_spinner(lottie_download, key="해시태그 생성"):
        time.sleep(5)
    st.balloons()

List=["A","B","C"]
selected_item = st.radio("Radio Part", List)
	
if selected_item == "A":
    st.write("A!!")
elif selected_item == "B":
    st.write("B!")
elif selected_item == "C":
    st.write("C!")

option = st.selectbox('Please select in selectbox!',
                    ('kyle', 'seongyun', 'zzsza'))
	
st.write('You selected:', option)

multi_select = st.multiselect('Please select somethings in multi selectbox!',
                                ['A', 'B', 'C', 'D'])
	
st.write('You selected:', multi_select)

add_selectbox = st.sidebar.selectbox("왼쪽 사이드바 Select Box", ("A", "B", "C"))

col1, col2, col3 = st.beta_columns(3)

with col1:
   st.header("A cat")
   st.image("https://static.streamlit.io/examples/cat.jpg", use_column_width=True)

