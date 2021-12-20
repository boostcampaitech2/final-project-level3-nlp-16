from tqdm import tqdm
import pandas as pd
import requests
import os
import urllib.request


def make_csv_from_bunjang(save_dir, categories, number_item):
    for category in categories:
        pages = int(number_item / 100 + 10)
        item_count = 0
        items = []
        
        for page in tqdm(range(pages)):
            pid_list = []
            url = "https://api.bunjang.co.kr/api/1/find_v2.json?f_category_id={}&page={}&order=date&req_ref=category&stat_device=w&n=100&version=4".format(
                category, page
            )
            response = requests.get(url)
            try:
                item_list = response.json()["list"]
                ids = [item["pid"] for item in item_list]
                pid_list.extend(ids)
            except:
                continue
            for pid in pid_list:
                if item_count == number_item:
                    break
                url = "https://api.bunjang.co.kr/api/1/product/{}/detail_info.json?version=4".format(
                    pid
                )
                response = requests.get(url)
                try:
                    details = response.json()["item_info"]
                    if (
                        details["keyword"] == ""
                        or details["description_for_detail"] == ""
                    ):
                        continue
                    details.pop("category_name")
                    details.pop("pay_option")
                    items.append(details)
                    item_count = item_count + 1
                except:
                    continue
            if item_count == number_item:
                break
        df = pd.DataFrame(items)
        bunjang_df = df[
            [
                "pid",
                "name",
                "keyword",
                "description_for_detail",
                "product_image",
            ]
        ]
        bunjang_df = bunjang_df.rename(
            {
                "name": "title",
                "keyword": "tag",
                "description_for_detail": "description",
                "product_image": "img",
            },
            axis="columns",
        )
        bunjang_df["origin_url"] = (
            "https://m.bunjang.co.kr/products/" + bunjang_df["pid"]
        )
        bunjang_df.to_csv(
            "{}/{}_data.csv".format(save_dir, category), header=True, index=True
        )


def download_csv_image(csv_dir, img_save_dir, categories):
    for category in categories:
        datas = pd.read_csv(csv_dir + "/{}_data.csv".format(category))
        try:
            if not os.path.exists(dir + "/img_data/{}".format(category)):
                os.makedirs(dir + "/img_data/{}".format(category))
        except:
            print("Error: Creating directory. " + dir)

        for i in tqdm(range(len(datas[["img", "pid"]]))):
            url = datas.iloc[i]["img"]
            pid = datas.iloc[i]["pid"]
            if not os.path.isfile(img_save_dir + "/{}/{}.jpg".format(category, pid)):
                urllib.request.urlretrieve(
                    url, img_save_dir + "/{}/{}.jpg".format(category, pid)
                )
