import es_api as es
from hanspell import spell_checker
import re

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
    for key,value in extracted_text[:3]:
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
    for doc in es_result['hits']['hits']: #res['hits']['hits']
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
        idf_values[key] *= len(re.findall(key.lower(), query.lower()))
    
    return idf_values

# def test_spelled_query(query):
#     spelled_query = spell_checker.check(query)

#     if spelled_query[0]:
#         es_result = es.es_search(spelled_query[2])
#     else:
#         es_result = es.es_search(query)

#     idf_values = get_idf_values(es_result)
#     if spelled_query[0]:
#         tf_idf_values = get_tf_idf_values(spelled_query[2], idf_values)
#     else:
#         tf_idf_values = get_tf_idf_values(query, idf_values)
#     extracted_spelled_text = sorted(tf_idf_values.items(), key=(lambda x: x[1]), reverse=True)

#     return extracted_spelled_text
