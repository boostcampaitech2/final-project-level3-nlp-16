from .elastic_search import es_api as es
from pprint import pprint as pp
import re


def extract_text(query):
    
    processed_query = preprocess_query(query)
    es_result = es.es_search(processed_query)
    idf_values = get_idf_values(es_result)
    tf_idf_values = get_tf_idf_values(processed_query, idf_values)

    extracted_text = sorted(tf_idf_values.items(), key=(lambda x: x[1]), reverse=True)
    
    return extracted_text

def preprocess_query(query):

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


if __name__ == '__main__':
    ex_query = "갤럭시 S8 64기가 상태굿 ㅡ중고폰3977 갤럭시s8 64기가 오키드그레이 상태굿 최초 LG개통했으나 모든유심다됩니다 선불.알뜰유심다됩니다 화면에 사진처럼 잔상조금있어 싸게판매합니다 ㅡ사진참조 터치및 모든 기능이상없이 작동잘됩니다 직거래는 인천 1호선 주안역입니다 택배거래도 가능합니다 이0ㅡ29오오ㅡ99이사"
    extract_text(ex_query)