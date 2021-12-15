import pandas as pd
import re

from datasets import load_dataset
from konlpy.tag import Okt
from transformers import AutoTokenizer, AutoModelForCausalLM
from numpy import dot
from numpy.linalg import norm
from nltk.tokenize import word_tokenize

MODEL = "nlprime/hash-tag-generation"


def main():
    ## Cleaning input data
    stop_word = load_dataset("psrpsj/stop_words")
    stop_word_list = stop_word["train"]["word"]

    title = "LG 15인치 노트북판매합니다"
    text = "lg 15u530-lh10k lg 울트라북 판매합니다 ssd 업글로 인터넷속도 부팅속도 빠릅니다 밧데리 약2시간이상갑니다 자세한 사양은 사진에있으니 참고하세요"
    exit_word = ""

    while exit_word != "exit":
        result = inference(title, text, stop_word_list)
        print("Hash tag result : ", result)
        exit_word = input("생성된 hash tag를 이용하시려면 exit를 입력하세요")


def inference(title, text, stop_word_list):

    tokenizer = AutoTokenizer.from_pretrained(MODEL, use_auth_token=True)

    model = AutoModelForCausalLM.from_pretrained(MODEL, use_auth_token=True)
    model.eval()

    # title
    title = re.sub(r"[^\w\s]", "", title)

    # text
    result = []
    text = text.replace("\n", " ")
    text = text.replace("\r", " ")
    text = re.sub(r"[^\w\s]", "", text)
    for word in word_tokenize(text):
        if word not in stop_word_list:
            result.append(word)
    text = " ".join(result)

    input_line = "<s>" + title + "<sep>" + text + "<sep>"

    input_ids = tokenizer.encode(
        input_line, add_special_tokens=True, return_tensors="pt"
    )

    output_sequence = model.generate(
        input_ids=input_ids,
        do_sample=True,
        max_length=512,
        num_return_sequences=1,
    )

    decode_output = tokenizer.decode(output_sequence[0], skip_special_tokens=True)
    decode_output = decode_output[len(input_line) + 1 :]
    decode_output = decode_output.split()
    # decode_output = list(set(decode_output))

    return decode_output


if __name__ == "__main__":
    main()
