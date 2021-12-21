import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"
import re
import pandas as pd
from tqdm import tqdm
from numpy import dot 
from numpy.linalg import norm

from pororo import Pororo
from transformers import AutoTokenizer

# PATH
get_path = os.getcwd().split('/')[:-3]
data_path =  '/' + os.path.join(*get_path) + '/data/raw_csv_data/'
file_list = os.listdir(data_path)

# Data Load
data = pd.DataFrame()
data_len = 0
for file in file_list:
    d = pd.read_csv(data_path + file, lineterminator='\n')
    d['category'] = file.split('_')[0]
    data_len += len(d)
    print(file, data_len)
    data = pd.concat([data, d])
    
print(data.columns)

def cos_sim(a, b): 
    return dot(a, b)/(norm(a)*norm(b))

def make_matrix(feats, list_data):
    freq_list = []
    for feat in feats:
        freq = 0 
        for word in list_data: 
            if feat == word: 
                freq += 1 
        freq_list.append(freq) 
    return freq_list

MODEL_NAME = 'klue/bert-base'
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print('Preprocessing start!!!')
input_data = []
for i in tqdm(range(10)):#len(data))):
    row = data.iloc[i]
    title_text = row['title']
    context_text = [d for d in row['description'].split('\n') if len(d) > 2]
    
    result_context = ''
    title = tokenizer.convert_ids_to_tokens(tokenizer(title_text)['input_ids'])[1:-1]
    for origin_text in context_text:
        context = tokenizer.convert_ids_to_tokens(tokenizer(origin_text)['input_ids'])[1:-1]
        
        title_vector = make_matrix(title, title)
        sim = make_matrix(title, context)
        similarity = cos_sim(title_vector, sim)

        if similarity > 0.3:
            result_context += origin_text
            
    if result_context:             
        input_data.append(result_context)

print('The number of original data : ', len(data))
print('The final number of data : ', len(input_data))

# NER Tagging
ner = Pororo(task="ner", lang='ko')
ner_vocab = []
for sentence in tqdm(input_data): 
    
    sentence_re = ' '.join(re.sub('[^A-Za-z0-9가-힣,/-:]', ' ', sentence).split(' '))
    
    pos_tag = ner(sentence_re[:500])
    tokens = [token for token, tag in pos_tag if tag == 'ARTIFACT' or tag == 'ORGANIZATION' or  tag == 'TERM' or tag == 'QUANTITY']
    ner_vocab.append(tokens)
    
    # print(sentence_re)
    # print(tokens)
    
# Postprocessing
print('Postprocessing start!!!')
output_vocab = []
for i, test in enumerate(ner_vocab):
    print(i, 'TEST :: ', test)
    hashtag = ''
    for word in test:
        
        m = re.search('.+원$|^[0-9, ]+$|[0-9]{4}|[일육칠팔구]+|대한통운|택배|coupang|쿠팡|배달의민족|배민|요기요|인터파크|농협|택|CU|cu|Cu|GS|gs|Gs|CJ|어린이집|학교|기숙사|공단|주민센터|아파트|아피트|대교|백화점|아울렛|우체국|지하철|건물|식당|반점|강서|강북|강변|남대문|수원|하남|신탄진|버스|게임|랜드|나라|월드|천국|당근|다나와|출|거리|층|만원|삼양|카카오톡|올리브영|cgv|스타벅스|넷플릭스|네프릭스|NEX|Netflix|티빙|네이버|유튜브|유투브|TUBE|YouTube|다이소|카톡|중고|경기|번개|코로나|바이러스|상위|하나|기스|지문|각각|기타|직|앞|뒤|팜|팝니다|만|총|평|출|천|등|이상|생겨|파라요|관절|총알|비비탄|세컨|방금|업무|개통|통신|아정당|반값|세상|사야|모든|삼형제|지방|통신|해지|순위|종류|이하|엄마|O1O|01O|O10|010|OIO|KAKA0|kakao|http|blog|naver|,|.+니다$|.+동$|.+위$|.+줄$|.+차$|.+당$|.+씩$|.+조$|.+호$|.+인$|.+분$|.+명$|.+씩$|.+번지$|.+길$|.+층$|.+선$|.+미터$|.+이내$|.+정도$|.+퍼$|.+개$|.+가지$|.+박스$|.+번$|.+땀$|.+장$|.+회$|.+퍼센트$|.+번째$|.+이상$|.+호선$|.+만$|.+짜리$|.+알리$', word)
        # 2:나이키,아이폰 / 3:삼성 / 4:안트라사이트 / 5:샤오미 
        
        if not m: 
            print(word)
            # hashtag += re.sub(' ', '', word) + ' ' # vocab_space.txt
            hashtag += '#' + word + ' ' # vocab_sharp.txt
        else:
            print('########### Delete!!!!!!', m.group())
            
    if len(hashtag) > 2:
        output_vocab.append(hashtag)

# Saveing vocab
save_path =  '/' + os.path.join(*get_path) + '/data/extraction_data/'
vocab_file_name = "vocab_sharp.txt"

f = open(save_path + vocab_file_name, 'w')
for v in output_vocab:
    f.write(v + '\n')
f.close()