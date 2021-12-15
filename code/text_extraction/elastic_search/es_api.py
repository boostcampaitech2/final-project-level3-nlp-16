import configparser
from typing import final
import requests
import json
from tqdm import tqdm
from pprint import pprint as pp
from elasticsearch import Elasticsearch
import os
import sys




def load_config():
    config = configparser.ConfigParser()
    # config.read('./../config.ini')
    
    user = 'elastic'
    password = "q391GEq6vSwrGf5Ykm0p67mB"
    endpoint = "https://auto-tag.es.asia-northeast3.gcp.elastic-cloud.com:9243"
    index_name = "auto_tag5"

    # auto_tag5 = 검색시 whitespace
    # auto_tag6 = 검색시 nori_tokenizer
    return user, password, endpoint, index_name


def es_search(query):
    es_user, es_password, es_endpoint, es_index_name = load_config()
    session = requests.Session()
    session.auth = (es_user, es_password)
    headers = {"Content-Type": "application/json; charset=utf-8"}
    body = {
        "query":{
            "match":{
                "vocab": query
            }
        },
        "explain": "true"
    }

    try:
        res = session.get(es_endpoint + "/" + es_index_name + "/_search?q=" + query, 
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers = headers, timeout=5)
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
            "number_of_shards": 5,
            "number_of_replicas": 1,
            "analysis": {
                "tokenizer": {
                    "korean_nori_tokenizer": {
                        "type": "nori_tokenizer",
                        "decompound_mode": "mixed",
                        "user_dictionary_rules": ["c++", "C샤프", "세종", "세종시 세종 시", "스마트티비 스마트 티비"]
                    }
                },
                "analyzer": {
                    "nori_analyzer": {
                        "type": "custom",
                        "tokenizer": "korean_nori_tokenizer",
                        "filter": ["nori_posfilter", "nori_readingform", "lowercase"]
                    },

                    "white_analyzer":{
                        "type": "custom",
                        "tokenizer": "whitespace",
                        "filter": ["lowercase"]
                    }
                },
                "filter": {
                    "nori_posfilter": {
                        "type": "nori_part_of_speech",
                        "stoptags": [
                            "E",
                            "IC",
                            "J",
                            "MAG",
                            "MM",
                            "NA",
                            "NR",
                            "SC",
                            "SE",
                            "SF",
                            "SH",
                            "SL",
                            "SN",
                            "SP",
                            "SSC",
                            "SSO",
                            "SY",
                            "UNA",
                            "UNKNOWN",
                            "VA",
                            "VCN",
                            "VCP",
                            "VSV",
                            "VV",
                            "VX",
                            "XPN",
                            "XR",
                            "XSA",
                            "XSN",
                            "XSV"
                        ]
                    }
                }
            }
        },
        "mappings": {
            "properties": {
                "vocab": {
                    "type": "text",
                    "analyzer": "white_analyzer",
                    "search_analyzer": "nori_analyzer"
                }
            }
        }
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
