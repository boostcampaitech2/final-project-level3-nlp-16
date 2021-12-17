# 당신도 중고 거래왕이 될 수 있습니다!

## Team Wrap-up Report
  * [notion]()

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
    - 1
    - 2
    - 3
  * Data
    - 1
    - 2
    - 3
  * Contributors
    * 김아경([github](https://github.com/EP000)): 
    * 김현욱([github](https://github.com/powerwook)): 
    * 김황대([github](https://github.com/kimhwangdae)): 
    * 박상류([github](https://github.com/psrpsj)): 
    * 정재현([github](https://github.com/JHyunJung)): 
    * 최윤성([github](https://github.com/choi-yunsung)): 

## Getting Started
  * Install requirements
    ``` bash
      # requirement 설치
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
│   ├── multimodal-clf
│   │   ├── multimodal_dataset.ipynb
│   │   └── train.py
│   ├── text_extraction
│   │   └── make_vocab.py
│   ├── text_generation
│   │   ├── arguments.py
│   │   └── train.py                  
│   ├── requirements.sh
│   └── server.py
│
└── data/                     
    ├── es_data/                        
    │   ├── pororo_ner_ver2.txt
    │   └── vocab.txt
    └── raw_csv_data/
        └── 600100001_data.csv
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
