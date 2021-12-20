import enum
import io
import os, sys 
import time
import requests
import pandas as pd
from PIL import Image

import streamlit as st
from streamlit_lottie import st_lottie, st_lottie_spinner

from transformers import AutoTokenizer, AutoModelForCausalLM

from models.mmclf.mmclf import MultimodalCLF, get_model, predict_from_multimodal, get_config, get_tokenizer
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from text_extraction.text_extraction import extract_text
from inference import cleaning, inference


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

    st.session_state.tokenizer = AutoTokenizer.from_pretrained(
        "nlprime/hash-tag-generator-small",use_auth_token=True
    )
    st.session_state.model = AutoModelForCausalLM.from_pretrained(
        "nlprime/hash-tag-generator-small",use_auth_token=True
    )
    
st.title("당신도 중고 거래왕이 될 수 있습니다!")

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

if custom_bg_img and title:
    with st_lottie_spinner(lottie_loading, height=100):
        st.session_state.labels = predict_from_multimodal(model=st.session_state.clf_model, 
                                                          tokenizer=st.session_state.clf_tokenizer, 
                                                          image_bytes=image_bytes, 
                                                          title=title, 
                                                          config=st.session_state.clf_config)


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
        
        
content=st.text_area("내용을 입력해주세요.", height = 150)

if st.button("해시태그 생성"):
    
    output = []
    input_text = title + ' ' + content
    extraction_tag = extract_text(input_text)

    output.extend(extraction_tag[:3])
    print('extraction_tag ', extraction_tag)
            
    # tokenizer = AutoTokenizer.from_pretrained(
    #     "nlprime/hash-tag-generator-small",use_auth_token=True
    # )
    # model = AutoModelForCausalLM.from_pretrained(
    #     "nlprime/hash-tag-generator-small",use_auth_token=True
    # )

    ids,max_len = cleaning(title, content, st.session_state.tokenizer)
    generation = inference(ids, max_len, st.session_state.model, st.session_state.tokenizer)
    print('generation ', generation)
    output.extend(generation)
    st.session_state.output = output
    
if st.session_state.get('output', None) is None:
    st.session_state.output = []

add_hashtag = st.text_input("추가로 입력할 해시태그를 적어주세요.")

output_str = ''
print(st.session_state.output)
for file in st.session_state.output:
    print(file)
    check_boxes = st.checkbox('#' + file)
    if check_boxes:
        # output_str = '#' + file
        output_str += '#' + file + ' '

if add_hashtag:
    for add_tag in add_hashtag.split(' '):
        # output_str += '#' + add_tag
        output_str += '#' + add_tag + ' '
        # st.write('#' + add_tag, end = ' ')

st.write(output_str)

# for t in test:
#     output_str += '#' + t + ' '
#     # st.write('#' + t, end = ' ')
print(output_str)

# for tag in output_str.split(' '):
#     st.write('#' + tag)