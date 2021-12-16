import streamlit as st
import time
import requests

from streamlit_lottie import st_lottie
from streamlit_lottie import st_lottie_spinner


st.set_page_config()
title=''
content=''
st.title("당신도 중고왕이 될 수 있습니다!")
title=st.text_input("제목을 입력해주세요.")
custom_bg_img = st.file_uploader(
    "상품 이미지를 올려주세요!", 
    type=["png", "jpg"]
)

def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_url_hello = "https://assets5.lottiefiles.com/packages/lf20_V9t630.json"
lottie_url_download = "https://assets7.lottiefiles.com/packages/lf20_wsy1p3ad.json"
lottie_hello = load_lottieurl(lottie_url_hello)
lottie_download = load_lottieurl(lottie_url_download)


if st.button("카테고리 예측"):
    with st_lottie_spinner(lottie_download, key="카테고리 예측"):
        time.sleep(5)
    st.balloons()


content=st.text_input("내용을 입력해주세요.")

if st.button("해시태그 생성"):
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

