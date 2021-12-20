import streamlit as st
import time
import requests
from transformers import AutoTokenizer, AutoModelForCausalLM

from streamlit_lottie import st_lottie
from streamlit_lottie import st_lottie_spinner
import pandas as pd
import re

import time
from inference import cleaning, inference
import io
from PIL import Image
import os
import sys

from streamlit_lottie import st_lottie, st_lottie_spinner

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

if "clf_tokenizer" not in st.session_state:
    st.session_state.clf_model = get_model()
    st.session_state.clf_tokenizer = get_tokenizer()
    st.session_state.clf_config = get_config()

st.title("당신도 중고 거래왕이 될 수 있습니다!")


content=st.text_input("내용을 입력해주세요.")
Result=[]
if st.button("해시태그 생성"):
    with st_lottie_spinner(lottie_download, key="해시태그 생성"):
        tokenizer = AutoTokenizer.from_pretrained(
        "nlprime/hash-tag-generator-small",use_auth_token=True
        )

        model = AutoModelForCausalLM.from_pretrained(
            "nlprime/hash-tag-generator-small",use_auth_token=True
        )
        ids,max_len=cleaning(title,content,tokenizer)
        Result = inference(ids,max_len,model,tokenizer)

    st.balloons()

    st.balloons()

selected_item = st.radio("Radio Part", Result)

custom_bg_img = st.file_uploader(
    "상품 이미지를 올려주세요!", 
    type=["png", "jpg", "jpeg"]
)

if custom_bg_img:
    image_bytes = custom_bg_img.getvalue()
    image = Image.open(io.BytesIO(image_bytes))
    st.image(image, caption='Uploaded Image', width=300)

title=st.text_input("상품 제목을 입력해주세요.")

if "labels" not in st.session_state:
    st.session_state.labels = []

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
if custom_bg_img and title:
    with st_lottie_spinner(lottie_loading, height=100):
        st.session_state.labels = predict_from_multimodal(model=st.session_state.clf_model, tokenizer=st.session_state.clf_tokenizer, image_bytes=image_bytes, title=title, config=st.session_state.clf_config)

if st.session_state.labels:
    st.write("추천 카테고리")
    cols = st.columns(5)
    for idx, col in enumerate(cols):
        bt = col.button(st.session_state.labels[idx])
        if bt:
            st.session_state.selected_idx = idx

    if "selected_idx" not in st.session_state:
        st.session_state.selected_idx = None

    if st.session_state.selected_idx != None:
        st.session_state.selected_category = st.selectbox(
            "카테고리를 선택해주세요",
            [st.session_state.labels[st.session_state.selected_idx]]
            +st.session_state.labels[:st.session_state.selected_idx]
            +st.session_state.labels[st.session_state.selected_idx+1:]
            )
    else:
        st.session_state.selected_category = st.selectbox("카테고리를 선택해주세요", st.session_state.labels)


    # col1, col2, col3, col4, col5 = st.columns(5)
    # bt1 = col1.button(st.session_state.labels[0])
    # bt2 = col2.button(st.session_state.labels[1])
    # bt3 = col3.button(st.session_state.labels[2])
    # bt4 = col4.button(st.session_state.labels[3])
    # bt5 = col5.button(st.session_state.labels[4])
    # for idx, bt in enumerate([bt1, bt2, bt3, bt4, bt5]):
    #     if bt:
    #         selected_category = st.selectbox(
    #             "카테고리를 선택해주세요", [st.session_state.labels[idx]]+st.session_state.labels[:idx]+st.session_state.labels[idx+1:])
    #         break
    # else:
    #     selected_category = st.selectbox("카테고리를 선택해주세요", st.session_state.labels)



content=st.text_area("내용을 입력해주세요.")

# if st.button("해시태그 생성"):
#     with st_lottie_spinner(lottie_download, key="해시태그 생성"):
#         time.sleep(5)
#     st.balloons()

# List=["A","B","C"]
# selected_item = st.radio("Radio Part", List)
	
# if selected_item == "A":
#     st.write("A!!")
# elif selected_item == "B":
#     st.write("B!")
# elif selected_item == "C":
#     st.write("C!")

# option = st.selectbox('Please select in selectbox!',
#                     ('kyle', 'seongyun', 'zzsza'))
	
# st.write('You selected:', option)

# multi_select = st.multiselect('Please select somethings in multi selectbox!',
#                                 ['A', 'B', 'C', 'D'])
	
# st.write('You selected:', multi_select)

# add_selectbox = st.sidebar.selectbox("왼쪽 사이드바 Select Box", ("A", "B", "C"))


# col1, col2, col3 = st.beta_columns(3)

# with col1:
#    st.header("A cat")
#    st.image("https://static.streamlit.io/examples/cat.jpg")

