import os
import re
import pandas as pd

from numpy import dot
from numpy.linalg import norm
from tqdm import tqdm
from transformers import AutoTokenizer

OUTPUT = "../../data/nlp_data"


def create_data_normal(path):
    print("Creating dataset with normal process")
    file_list = os.listdir(path)
    target_list = [csv_file for csv_file in file_list if csv_file.endswith(".csv")]
    total = []
    for target in tqdm(target_list):
        data = pd.read_csv(os.path.join(path, target))
        for index in range(len(data)):
            title = data.iloc[index]["title"]
            text = data.iloc[index]["description"]
            tag = data.iloc[index]["tag"]

            title = re.sub(r"[^\w\s]", "", title)
            text = text.replace("\n", " ")
            text = text.replace("\r", " ")
            text = re.sub(r"[^\w\s]", "", text)
            tag = tag.replace(",", " ")
            tag = re.sub(r"[^\w\s]", "", tag)
            tag = tag.replace(" ", "<sep>")
            line = "<s>" + title + "<sep>" + text + "<sep>" + tag + "</s>"
            tmp = {"data": line}
            total.append(tmp)
    dataset = pd.DataFrame(total)
    dataset.to_csv(os.path.join(OUTPUT, "total_text_tag.csv"))
    return dataset


def create_data_cosine(path, cosine_rate):
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
    file_list = os.listdir(path)
    target_list = [csv_file for csv_file in file_list if csv_file.endswith(".csv")]
    total = []
    tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")
    for target in tqdm(target_list):
        target_file = pd.read_csv(os.path.join(path, target))
        for idx in range(len(target_file)):
            title = target_file.iloc[idx]["title"]
            text = target_file.iloc[idx]["description"]
            tag = target_file.iloc[idx]["tag"]

            title = re.sub(r"[^\w\s]", "", title)

            # calculate cosine
            text_line = [d for d in text.split("\n") if len(d) > 2]
            title_token = tokenizer.convert_ids_to_tokens(tokenizer.encode(title))[1:-1]
            line_list = []
            for line in text_line:
                line_token = tokenizer.convert_ids_to_tokens(tokenizer.encode(line))[
                    1:-1
                ]
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
                tag = tag.replace(" ", "<sep>")
                result = "<s>" + title + "<sep>" + text + "<sep>" + tag + "</s>"
                tmp = {"data": result}
                total.append(tmp)

    dataset = pd.DataFrame(total)
    dataset.to_csv(os.path.join(OUTPUT, "cosine_result.csv"))
    return dataset
