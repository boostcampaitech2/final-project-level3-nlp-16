from fastapi import FastAPI
from text_extraction import text_extraction as te
app = FastAPI()


@app.get("/extraction")
async def text_extraction(q: str):
    # "갤럭시 S8 64기가 상태굿 ㅡ중고폰3977 갤럭시s8 64기가 오키드그레이 상태굿 최초 LG개통했으나 모든유심다됩니다 선불.알뜰유심다됩니다 화면에 사진처럼 잔상조금있어 싸게판매합니다 ㅡ사진참조 터치및 모든 기능이상없이 작동잘됩니다 직거래는 인천 1호선 주안역입니다 택배거래도 가능합니다 이0ㅡ29오오ㅡ99이사"
    result = te.extract_text(q)
    
    l = []
    for key, val in result:
        l.append("#{}".format(key))

    return l