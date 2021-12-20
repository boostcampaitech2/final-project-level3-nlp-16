import re

from transformers import AutoTokenizer, AutoModelForCausalLM
import time

MODEL = "nlprime/hash-tag-generator"


def main():

    tokenizer = AutoTokenizer.from_pretrained(MODEL, use_auth_token=True)

    model = AutoModelForCausalLM.from_pretrained(MODEL, use_auth_token=True)

    title = "파손폰매입 중고폰매입 아이폰12 아이폰11 아이폰xs 아이폰x"
    text = "👉 중고폰 파손폰 폐폰 태블릿 전부 매입합니다  👈 👉 허위가격제시.불필요한 차감 절대 하지않아요 👈 👉 부평지하상가 매장 & 번개장터 7년차 운영중이에요 👈 😁 24시간 톡 . 전화 . 문자 언제든 상담 환영해요 😁 😁 아이폰사설수리점도 운영하고 있으니 . 완파된 폰도 전부 점검및 매입 가능합니다 😁😁 부평아이클리닉 😁 검색해서 보시면 방문자 실리뷰및 매장 확인가능하오니 안심하셔도 됩니다. 🤩 거래방법 > 직거래시 매장방문 해주시면 되시며.. 또는 택배거래 진행하고있습니다. 택배거래시 상점 찜 팔로우시 택배비 3천원 지원해드리고 있습니다 🤩 👉 택배거래시 케이스.필름.박스 전부 폐기하오니 참고해주세요👈 ✌ 매장방문 > 부평역지하상가 분수대 오셔서 연락주시면 빠르세요✌ 택배거래 > 인천 부평구 광장로16 민자역사쇼핑몰 지하2층20호 매주 화요일은 휴무이오니 참고부탁드리겠습니다 👊 상호 > 두리모바일 👊 👊  택배거래시 액정및기능이상 유무 꼭 확인부탁드릴께요 🤗"

    ids, max_len = cleaning(title, text, tokenizer)
    exit_word = ""
    while exit_word != "exit":
        start = time.time()
        result = inference(ids, max_len, model, tokenizer)
        print("Hash tag result : ", result)
        end = time.time()
        print(end - start)
        exit_word = input("생성된 hash tag를 이용하시려면 exit를 입력하세요")


def cleaning(title, text, tokenizer):
    # title
    title = re.sub(r"[^\w\s]", " ", title)

    # text
    text = text.replace("\n", " ")
    text = text.replace("\r", " ")
    text = re.sub(r"[^\w\s]", "", text)
    input_line = "<s>" + title + "<sep>" + text + "<sep>"

    input_ids = tokenizer.encode(
        input_line, add_special_tokens=True, return_tensors="pt"
    )

    return input_ids, len(input_line)


def inference(input_ids, max_len, model, tokenizer):

    output_sequence = model.generate(
        input_ids=input_ids,
        do_sample=True,
        max_length=max_len,
        num_return_sequences=1,
        top_p=0.1,
    )

    decode_output = tokenizer.decode(output_sequence[0], skip_special_tokens=True)
    decode_output = decode_output[max_len + 1 :]
    decode_output = decode_output.split(",")
    decode_output = list(set(decode_output))
    decode_output = list(filter(None, decode_output))
    return decode_output


if __name__ == "__main__":
    main()
