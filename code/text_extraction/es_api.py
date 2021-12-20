import configparser
import requests
import json
from tqdm import tqdm
from pprint import pprint as pp
from elasticsearch import Elasticsearch


def load_config():
    config = configparser.ConfigParser()
    config.read(".config.ini")

    user = config["ES"]["USER"]
    password = config["ES"]["PASSWORD"]
    endpoint = config["ES"]["ENDPOINT"]
    index_name = "auto_tag"

    return user, password, endpoint, index_name


def es_search(query):
    es_user, es_password, es_endpoint, es_index_name = load_config()
    session = requests.Session()
    session.auth = (es_user, es_password)

    try:
        res = session.get(
            es_endpoint
            + "/"
            + es_index_name
            + "/_search?q={}&explain=true&size=10000".format(query)
        )
    except requests.exceptions.RequestException as erra:
        print("es_search() Exception : ", erra)
    except requests.exceptions.HTTPError as hter:
        print("http Exception : ", hter)

    return res.json()


def load_vocab_data(file_path):
    with open(file_path, "r", encoding="utf8") as file:
        vocab_list = []
        vocabs = file.readlines()
        for vocab in vocabs:
            vocab = vocab.replace("\n", "")
            vocab = vocab.replace("  ", "")
            vocab_list.append(vocab)

    return vocab_list


def es_indexing(file_path):

    vocab_list = load_vocab_data(file_path)
    es_user, es_password, es_endpoint, es_index_name = load_config()

    headers = {"Content-Type": "application/json; charset=utf-8"}
    session = requests.Session()
    session.auth = (es_user, es_password)
    for index, s in enumerate(tqdm(vocab_list)):
        body = {"vocab": s}
        try:
            res = session.put(
                es_endpoint + "/" + es_index_name + "/_doc/" + str(index + 1),
                data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
                headers=headers,
            )
        except requests.exceptions.RequestException as erra:
            print("es_indexing() Exception : ", erra)

    print("ElasticSearch Data Indexing Finished")


def es_make_index():
    es_user, es_password, es_endpoint, es_index_name = load_config()

    index_settings = {
        "settings": {
            "number_of_shards": 2,
            "number_of_replicas": 1,
            "analysis": {
                "analyzer": {
                    "white_analyzer": {
                        "type": "custom",
                        "tokenizer": "whitespace",
                        "filter": ["lowercase"],
                    }
                }
            },
        },
        "mappings": {
            "properties": {
                "vocab": {
                    "type": "text",
                    "analyzer": "white_analyzer",
                    "search_analyzer": "white_analyzer",
                }
            }
        },
    }
    headers = {"Content-Type": "application/json; charset=utf-8"}
    session = requests.Session()
    session.auth = (es_user, es_password)
    try:
        res = session.put(
            es_endpoint + "/" + es_index_name,
            data=json.dumps(index_settings, ensure_ascii=False).encode("utf-8"),
            headers=headers,
        )
    except requests.exceptions.RequestException as erra:
        print("es_make_index() Exception : ", erra)

    print(res)
    print("ElasticSearch make Index Finished")
