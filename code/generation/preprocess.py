import os
import re
import pandas as pd

from datasets import load_dataset
from numpy import dot
from numpy.linalg import norm
from tqdm import tqdm
from transformers import AutoTokenizer


def create_preprocess_data(data_args):
    dataset = load_dataset(data_args.dataset_name, use_auth_token=True)
    dataset = dataset["train"]

    if data_args.preprocess_type == "normal":
        return create_data_normal(dataset, data_args.dataset_output_dir)
    elif data_args.preprocess_type == "cosine":
        return create_data_cosine(
            dataset, data_args.dataset_output_dir, data_args.cosine_rate
        )
    else:
        raise ValueError("DataTrainingArguemnts preprocess_type value error!")


def create_data_normal(dataset, output_dir):
    print("Creating dataset with normal process")
    total = []
    for idx in tqdm(range(len(dataset))):
        title = dataset[idx]["title"]
        text = dataset[idx]["description"]
        tag = dataset[idx]["tag"]

        title = re.sub(r"[^\w\s]", "", title)
        text = text.replace("\n", " ")
        text = text.replace("\r", " ")
        text = re.sub(r"[^\w\s]", "", text)
        tag = tag.replace(",", " ")
        tag = re.sub(r"[^\w\s]", "", tag)
        tag = tag.replace(" ", ",")
        line = "<s>" + title + "<sep>" + text + "<sep>" + tag + "</s>"
        tmp = {"data": line}
        total.append(tmp)

    return_dataset = pd.DataFrame(total)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return_dataset.to_csv(os.path.join(output_dir, "normal_preprocess.csv"))
    return return_dataset


def create_data_cosine(dataset, output_dir, cosine_rate):
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
    tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")

    for idx in tqdm(range(len(dataset))):
        title = dataset[idx]["title"]
        text = dataset[idx]["description"]
        tag = dataset[idx]["tag"]

        title = re.sub(r"[^\w\s]", "", title)

        # calculate cosine
        text_line = [d for d in text.split("\n") if len(d) > 2]
        title_token = tokenizer.convert_ids_to_tokens(tokenizer.encode(title))[1:-1]
        line_list = []
        for line in text_line:
            line_token = tokenizer.convert_ids_to_tokens(tokenizer.encode(line))[1:-1]
            title_matrix = create_matrix(title_token, title_token)
            title_line_matrix = create_matrix(title_token, line_token)
            cosine_sim = cal_cosine_sim(title_matrix, title_line_matrix)

            # if able to accept, preprocess the sentence and add to list
            if cosine_sim > cosine_rate:
                line = line.replace("\n", " ")
                line = line.replace("\r", " ")
                line = re.sub(r"[^\w\s]", "", line)
                if not line.startswith("http"):
                    line_list.append(line)

        if len(line_list) > 0:
            text = " ".join(line_list)
            tag = tag.replace(",", " ")
            tag = re.sub(r"[^\w\s]", "", tag)
            tag = tag.replace(" ", ",")
            result = "<s>" + title + "<sep>" + text + "<sep>" + tag + "</s>"
            tmp = {"data": result}
            total.append(tmp)

    return_dataset = pd.DataFrame(total)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return_dataset.to_csv(os.path.join(output_dir, "cosine_preprocess.csv"))
    return return_dataset
