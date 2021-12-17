import os
import re
import pandas as pd
import nltk

from datasets import load_dataset
from numpy import dot
from numpy.linalg import norm
from nltk.tokenize import word_tokenize
from konlpy.tag import Okt
from tqdm import tqdm


def create_preprocess_data(data_args):
    nltk.download("punkt")
    dataset = load_dataset(data_args.dataset_name, use_auth_token=True)
    dataset = dataset["train"]
    stop_word = load_dataset("psrpsj/stop_words")
    stop_word_list = stop_word["train"]["word"]

    if data_args.preprocess_type == "normal":
        return create_data_normal(dataset, data_args.dataset_output_dir)
    elif data_args.preprocess_type == "cosine":
        return create_data_cosine(
            dataset, data_args.dataset_output_dir, data_args.cosine_rate, stop_word_list
        )
    else:
        raise ValueError("DataTrainingArguemnts preprocess_type value error!")


def cleaning(title, txt, hash_tag, stop_word_list):
    ## title
    title = re.sub(r"[^\w\s]", "", title)

    ## txt
    txt = txt.replace("\r", " ")
    txt_list = txt.split("\n")
    result = []
    for text in txt_list:
        text = re.sub(r"[^\w\s]", "", text)
        text = text.lstrip()
        after_stop = []
        for word in word_tokenize(text):
            if word not in stop_word_list:
                after_stop.append(word)
        text = " ".join(after_stop)
        if not text.lower().startswith(("http", "www", "번개페이")):
            result.append(text)
    txt_list = list(filter(None, result))

    ## hash
    hash_tag = hash_tag.replace(",", " ")
    hash_tag = hash_tag.replace("\n", "")
    hash_tag = hash_tag.replace("\r", "")
    hash_tag = re.sub(r"[^\w\s]", "", hash_tag)
    hash_tag = hash_tag.replace(" ", ",")
    return title, txt_list, hash_tag


def create_data_normal(dataset, output_dir, stop_word_list):
    print("Creating dataset with normal process")
    total = []
    for idx in tqdm(range(len(dataset))):
        title = dataset[idx]["title"]
        text = dataset[idx]["description"]
        tag = dataset[idx]["tag"]

        title, text_list, tag = cleaning(title, text, tag, stop_word_list)
        text = " ".join(text_list)
        line = "<s>" + title + "<sep>" + text + "<sep>" + tag + "</s>"
        tmp = {"data": line}
        total.append(tmp)

    return_dataset = pd.DataFrame(total)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return_dataset.to_csv(os.path.join(output_dir, "normal_preprocess.csv"))
    return return_dataset


def create_data_cosine(dataset, output_dir, cosine_rate, stop_word_list):
    """Create dataset using Cosine Similarity"""

    def create_matrix(input1, input2):
        matrix = []
        for single_input in input1:
            occurence = input2.count(single_input)
            matrix.append(occurence)
        return matrix

    def cal_cosine_sim(matrix1, matrix2):
        return dot(matrix1, matrix2) / (norm(matrix1) * norm(matrix2))

    print("Creating dataset with Cosine Similarity Process")
    total = []
    tokenizer = Okt()

    for idx in tqdm(range(len(dataset))):
        title = dataset[idx]["title"]
        text = dataset[idx]["description"]
        tag = dataset[idx]["tag"]

        title, text_line, tag = cleaning(title, text, tag, stop_word_list)

        # calculate cosine
        title_token = tokenizer.morphs(title)
        title_matrix = create_matrix(title_token, title_token)
        line_list = []
        for line in text_line:
            line_token = tokenizer.morphs(line)
            title_line_matrix = create_matrix(title_token, line_token)
            cosine_sim = cal_cosine_sim(title_matrix, title_line_matrix)

            # if able to accept, preprocess the sentence and add to list
            if cosine_sim > cosine_rate:
                line_list.append(line)

        if len(line_list) > 0:
            text = ". ".join(line_list)
            result = "<s>" + title + "<sep>" + text + "<sep>" + tag + "</s>"
            tmp = {"data": result}
            total.append(tmp)

    return_dataset = pd.DataFrame(total)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return_dataset.to_csv(os.path.join(output_dir, "cosine_preprocess.csv"))
    return return_dataset
