import pandas as pd
import re

from datasets import load_dataset
from konlpy.tag import Okt
from transformers import AutoTokenizer, GPT2LMHeadModel
from numpy import dot
from numpy.linalg import norm
from nltk.tokenize import word_tokenize

MODEL = "skt/kogpt2-base-v2"


def inference(title, text):

    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=MODEL,
        max_len=1024,
        padding="max_length",
        add_special_tokens=True,
        return_tensors="pt",
        truncation=True,
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>",
        sep_token="<sep>",
    )

    model = GPT2LMHeadModel.from_pretrained(MODEL)
    model.eval()

    ## Cleaning input data
    stop_word = load_dataset("psrpsj/stop_words")
    stop_word_list = stop_word["train"]["word"]

    # title
    title = re.sub(r"[^\w\s]", "", title)

    # text
    text_list = text.split(".")
    result = []
    for sentence in text_list:
        sentence = re.sub(r"[^\w\s]", "", sentence)
        sentence = sentence.lstrip()
        stop_process = []
        for word in word_tokenize(sentence):
            if word not in stop_word_list:
                stop_process.append(word)
        text = " ".join(stop_process)
        if not text.lower().startswith(("http", "www", "번개페이")):
            result.append(text)
    result = list(filter(None, result))
    text = " ".join(result)

    input_line = "<s>" + title + "<sep>" + text + "<sep>"
