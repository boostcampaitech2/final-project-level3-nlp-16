import configparser
import requests
import json
from tqdm import tqdm
from pprint import pprint as pp

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath("__file__")))))))

from elasticsearch import Elasticsearch
from config_file_read import read_config

def load_config():
    
    config = read_config()
    
    user = config['ES']['USER']
    password = config['ES']['PASSWORD']
    endpoint = config['ES']['ENDPOINT']
    index_name = config['ES']['INDEX']

    return user, password, endpoint, index_name


class ESApi:
    def __init__(self):
        self.es_user, self.es_password, self.es_endpoint, self.index_name = load_config()
        self.es = Elasticsearch(hosts = self.es_endpoint, http_auth=(self.es_user, self.es_password))
    
    def load_elastic(self):
        return self.es
    
    def es_search(self, query):
        session = requests.Session()
        session.auth = (self.es_user, self.es_password)
        try:
            res = session.get(self.es_endpoint+"/_search?q="+query, timeout = 5)
        except requests.exceptions.RequestException as erra:
            print("es_search() Exception : ", erra)
        
        return res.json()['hits']['hits']

    def es_indexing(self):

        with open("/Users/jhyun/Git/final-project-level3-nlp-16/data/es_data/pororo_ner_ver2.txt", "r", encoding='utf8') as file:
            renew = []
            strings = file.readlines()
            for s in strings:
                s = s.replace("\n", "")
                s = s.replace("  ", "")
                s = s.replace("\"", " ")
                renew.append(s)
        headers = {'Content-Type': 'application/json; charset=utf-8'}

        session = requests.Session()
        session.auth = (self.es_user, self.es_password)

        for index, s in enumerate(renew):
            body = {"vocab" : s}
            try:
                res = session.put(self.es_endpoint+"/"+self.index_name+"/_doc/" + str(index+1), data = json.dumps(body,ensure_ascii=False).encode('utf-8'), headers=headers)   
            except requests.exceptions.RequestException as erra:
                print("es_indexing() Exception : ", erra)
            print(res)

    def es_make_index(self):
        try:
            res = requests.put(self.es_endpoint+"/" + self.index_name, timeout = 5)
        except requests.exceptions.RequestException as erra:
            print("es_make_index() Exception : ", erra)



