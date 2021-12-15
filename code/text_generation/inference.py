import pandas as pd
import re

from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = "nlprime/hash-tag-generation"


def main():

    tokenizer = AutoTokenizer.from_pretrained(MODEL, use_auth_token=True)

    model = AutoModelForCausalLM.from_pretrained(
        "/opt/ml/final-project-level3-nlp-16/code/text_generation/output"
    )

    title = "LG 15인치 노트북판매합니다"
    text = "lg 15u530-lh10k lg 울트라북 판매합니다 ssd 업글로 인터넷속도 부팅속도 빠릅니다 밧데리 약2시간이상갑니다 자세한 사양은 사진에있으니 참고하세요"

    ids, max_len = cleaning(title, text, tokenizer)
    exit_word = ""
    while exit_word != "exit":
        result = inference(ids, max_len, model, tokenizer)
        print("Hash tag result : ", result)
        exit_word = input("생성된 hash tag를 이용하시려면 exit를 입력하세요")


def cleaning(title, text, tokenizer):
    # title
    title = re.sub(r"[^\w\s]", "", title)

    # text
    result = []
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
        top_p=0.6,
    )

    decode_output = tokenizer.decode(output_sequence[0], skip_special_tokens=True)
    decode_output = decode_output[max_len + 1 :]
    decode_output = decode_output.split(",")
    decode_output = list(set(decode_output))
    decode_output = list(filter(None, decode_output))

    return decode_output


if __name__ == "__main__":
    main()
