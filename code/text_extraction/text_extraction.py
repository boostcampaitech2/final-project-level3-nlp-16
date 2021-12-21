import text_extraction.es_api as es 
from hanspell import spell_checker
import re
import math 

def extract_text(query):
    
    processed_query = preprocess_query(query)
    spelled_query = check_query_spell(processed_query)
    es_result = es.es_search(spelled_query)
    idf_values = get_idf_values(es_result)
    tf_idf_values = get_tf_idf_values(spelled_query, idf_values)
    extracted_text = sorted(tf_idf_values.items(), key=(lambda x: x[1]), reverse=True)
    result = make_result(extracted_text)
    
    return result

def make_result(extracted_text):
    result = []
    for key, value in extracted_text[:3]:
        result.append(key)
    return result

def check_query_spell(query):
    spelled_query = spell_checker.check(query)
    if spelled_query[0]:
        return spelled_query[2]
    return query

def preprocess_query(query):

    query = re.sub(r"[^\w\s]", "", query)
    query = query.replace("\n", " ")
    query = query.replace("\r", " ")
    query = query.replace(",", " ")
    query = query.replace(".", " ")

    return query

def get_idf_values(es_result):
    idf_values = {}
    er = []
    for doc in es_result['hits']['hits']: 
        for term in doc["_explanation"]["details"]:
            try:
                word = term["description"]
                word = word.split(' ')[0][13:]
                if word not in idf_values:
                    idf_value = term['details'][0]['details'][1]['value'] 
                    idf_values[word] = idf_value
            except IndexError as error:
                er.append(term)

    return idf_values

def get_tf_idf_values(query, idf_values):
    for key in idf_values:
        idf_values[key] *= math.log1p(len(re.findall(key.lower(), query.lower())))
    
    return idf_values
