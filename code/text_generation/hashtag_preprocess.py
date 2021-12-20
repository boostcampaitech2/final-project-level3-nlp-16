import pandas as pd
import argparse

from datasets import load_dataset
from pororo import Pororo
from tqdm import tqdm


def preprocess(start_idx, end_idx):
    dataset = load_dataset("nlprime/secondhand-goods", use_auth_token=True)
    dataset = dataset["train"]
    ner = Pororo(task="ner", lang="ko")
    result = []
    error_idx = []
    for idx in tqdm(range(start_idx, end_idx)):
        try:
            hashtag = dataset["tag"][idx]
            hashtag_ner_list = ner(hashtag)
            for hashtag_ner in hashtag_ner_list:
                if hashtag_ner[1] == "CITY" or hashtag_ner[1] == "LOCATION":
                    hashtag = hashtag.replace(hashtag_ner[0], "")
                    print(hashtag_ner[0])
            tmp = {"index": idx, "pid": dataset["pid"][idx], "tag": hashtag}
            result.append(tmp)
        except BaseException as e:
            error_idx.append({"index": idx, "error": e})
    result_pd = pd.DataFrame(result)
    error_pd = pd.DataFrame(error_idx)

    result_pd.to_csv(
        "./preprocess_" + str(start_idx) + "_" + str(end_idx) + ".csv", index=False
    )
    error_pd.to_csv(
        "./preprocess_" + str(start_idx) + "_" + str(end_idx) + "_" + "error.csv",
        index=False,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--start_idx", type=int, required=True)
    parser.add_argument("--end_idx", type=int, required=True)

    args = parser.parse_args()
    preprocess(args.start_idx, args.end_idx)
