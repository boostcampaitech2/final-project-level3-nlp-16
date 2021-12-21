# 당신도 중고 거래왕이 될 수 있습니다!

## Table of Contents
  1. [Project Overview](#Project-Overview)
  2. [Getting Started](#Getting-Started)
  3. [Hardware](#Hardware)
  4. [Code Structure](#Code-Structure)
  5. [Detail](#Detail)

## Project Overview
  * 목표
    1. 멀티모달 분류모델을 활용하여 입력된 상품 이미지와 제목으로 카테고리 분류
    2. 생성/추출모델을 통해 상품 노출 빈도를 높일 수 있는 해시태그 생성
  * 모델
    1. EfficientNet-b0 와 BERT Classifier 모델을 이용한 카테고리 분류모델
    2. Elastic Search 와 TF-IDF를 이용한 HashTag 추출모델
    3. [skt/kogpt-base-v2](https://github.com/SKT-AI/KoGPT2)를 기반한 데이터 fine-tuned HashTag 생성모델
  * Data
    - 1. 번개장터 crawling 데이터 (분야 : 전가기기)

  * Contributors
    * 김아경([github](https://github.com/EP000)): 추출모델설계, Text 데이터 전처리
    * 김현욱([github](https://github.com/powerwook)): 이미지 데이터 전처리, 분류모델 검증
    * 김황대([github](https://github.com/kimhwangdae)): 생성모델 설계, Streamlit 설계
    * 박상류([github](https://github.com/psrpsj)): 생성모델 설계, Text 데이터 전처리
    * 정재현([github](https://github.com/JHyunJung)): 데이터 크롤링, Elastic Search 설계 및 구현
    * 최윤성([github](https://github.com/choi-yunsung)): Project Manager, 분류모델 설계

## Getting Started
  * Install requirements
    ``` bash
      # requirement 설치
      cd code
      pip install -r requirements.txt 
    ```
## Hardware
The following specs were used to create the original solution.
- Ubuntu 18.04.5 LTS
- Intel(R) Xeon(R) Gold 5120 CPU @ 2.20GHz
- NVIDIA Tesla V100-SXM2-32GB

## Code Structure
```text
├── code/                   
│   ├── crawl
│   │   └── bunjang_crawl.py
│   │
│   ├── multimodal-clf
│   │   ├── configs
│   │   │   ├── data/secondhad-goods.yaml
│   │   │   └── model/mobilenetv3_kluebert.yaml
│   │   ├── src
│   │   │   ├── augmentation
│   │   │   │   ├── methods.py
│   │   │   │   ├── policies.py
│   │   │   │   └── transforms.py
│   │   │   ├── utils
│   │   │   │   ├── common.py
│   │   │   │   └── data.py
│   │   │   ├── dataloader.py
│   │   │   ├── model.py
│   │   │   └── traniner.py
│   │   └── train.py
│   │   
│   ├── prototype
│   │   ├── models/mmclf
│   │   │   ├── best.py
│   │   │   ├── config.yaml
│   │   │   ├── mmclf.py
│   │   │   ├── special_tokens_map.json
│   │   │   ├── tokenizer_config.json
│   │   │   ├── tokenizer.json
│   │   │   └── vocab.txt
│   │   ├── app.py
│   │   └── inference.py
│   │   
│   ├── text_extraction
│   │   ├── es_api.py
│   │   ├── make_vocab.py
│   │   └── text_extraction.py
│   │
│   ├── text_generation
│   │   ├── arguments.py
│   │   ├── data.py
│   │   ├── hashtag_preprocess.py
│   │   ├── inference.py
│   │   ├── preprocess.py
│   │   └── train.py                  
│   │
│   ├── requirements.txt
│   └── README.md
│
└── data/es_data                     
    └── vocab_space_ver2.txt                        
    
```
## Detail
  * 멀티모달 분류모델을 활용하여 입력된 상품 이미지와 제목으로 카테고리 분류
    * 1
    * 2
    * 3
    
  * 생성/추출모델을 통해 상품 노출 빈도를 높일 수 있는 해시태그 생성
    * 1
    * 2
    * 3
