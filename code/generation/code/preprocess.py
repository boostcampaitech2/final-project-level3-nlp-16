import os
import re
import pandas as pd

from tqdm import tqdm

OUTPUT = "../data/nlp_data"


def create_tagset(path):
    path_data = pd.read_csv(path)
    total = []
    for tag_line in path_data["tag"]:
        tag_line = tag_line.replace(",", " ")
        tag_line = re.sub(r"[^\w\s]", "", tag_line)
        tag_list = tag_line.split()
        for tag in tag_list:
            tmp = {"tag": tag}
            total.append(tmp)
    dataset = pd.DataFrame(total)
    new_file = os.path.basename(path)
    new_file = new_file.replace(".csv", "")
    new_file += "_tag.csv"
    if not os.path.exists(OUTPUT):
        os.mkdir(OUTPUT)
    dataset.to_csv(os.path.join(OUTPUT, new_file))


def create_tagwholeset(path):
    file_list = os.listdir(path)
    target_list = [csv_file for csv_file in file_list if csv_file.endswith(".csv")]
    total = []
    for target in tqdm(target_list):
        data = pd.read_csv(os.path.join(path, target), encoding="UTF8")
        for tag_line in data["tag"]:
            tag_line = tag_line.replace(",", " ")
            tag_line = re.sub(r"[^\w\s]", "", tag_line)
            tag_list = tag_line.split()
            tag_list = list(set(tag_list))
            for tag in tag_list:
                tmp = {"tag": "<s>" + tag + "</s>"}
                total.append(tmp)
    dataset = pd.DataFrame(total)
    if not os.path.exists(OUTPUT):
        os.makedirs(OUTPUT)
    dataset.to_csv(os.path.join(OUTPUT, "total_tag.csv"))
    return dataset


def create_data_with_passage(path):
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


def create_data_for_all_data(path="/opt/ml/data/nlp_data/all_data.csv"):
    data = pd.read_csv(path, lineterminator="\n")
    total = []
    for index in tqdm(range(len(data))):
        title = data.iloc[index]["title"]
        text = data.iloc[index]["context"]
        tag = data.iloc[index]["hashtag"]

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
    dataset.to_csv(os.path.join(OUTPUT, "total_all_data.csv"))
    return dataset
