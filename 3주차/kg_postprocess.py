# -*- coding: utf-8 -*-
"""
kg_postprocess.py
──────────────────────────────────────────────────────────────────────────────
입력:  triples.csv   (subject,predicate,object,src_url)
출력: triples_clean.csv (subject_id,predicate,object_clean,src_url)

📌 기능
    1) **subject 중복 해결**  → src_url 의 MD5 해시 8자를 subject_id 로 사용
    2) **object 정제**        → Okt 형태소 분석 후 2~4자 명사 추출, 중복 제거, 최대 3개
    3) **predicate 재매핑**   → object·subject 키워드에 기반한 간단 룰 (origin_of, subtype_of, ...)
    4) CSV 저장 + 요약 출력

필수:  pip install pandas konlpy sklearn tqdm
예시:   python kg_postprocess.py triples.csv triples_clean.csv
"""

import sys, csv, hashlib, re, pathlib
import pandas as pd
from konlpy.tag import Okt
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

if len(sys.argv) < 3:
    sys.exit("Usage: python kg_postprocess.py <input_csv> <output_csv>")

IN_CSV, OUT_CSV = map(pathlib.Path, sys.argv[1:3])

df = pd.read_csv(IN_CSV)
okt = Okt()

# 1) subject_id (url hash)
df["subject_id"] = df["src_url"].apply(lambda u: hashlib.md5(u.encode()).hexdigest()[:8])

# 2) object 정제 함수
def clean_object(text: str) -> str:
    nouns = [n for n in okt.nouns(str(text)) if 2 <= len(n) <= 4]
    nouns = list(dict.fromkeys(nouns))          # 중복 제거 (order keep)
    return " ".join(nouns[:3])

df["object_clean"] = df["object"].apply(clean_object)

# 3) predicate 재매핑 룰
rules = [
    (re.compile(r"유래|기원|발생"), "origin_of"),
    (re.compile(r"구분|나뉘"), "subtype_of"),
    (re.compile(r"반대|상반"), "opposite_to"),
    (re.compile(r"예"), "example_of"),
    (re.compile(r"다르게|구별"), "contrast_with"),
    (re.compile(r"포함|속하"), "includes"),
]

def remap_pred(row):
    txt = f"{row['subject']} {row['object']}"
    for pat, tag in rules:
        if pat.search(txt):
            return tag
    return row["predicate"] or "defines"

df["predicate_new"] = df.apply(remap_pred, axis=1)

# 4) 중복 제거
final_cols = ["subject_id", "predicate_new", "object_clean", "src_url"]
clean_df = df[final_cols].drop_duplicates()

clean_df.columns = ["subject", "predicate", "object", "src_url"]
clean_df.to_csv(OUT_CSV, index=False, quoting=csv.QUOTE_MINIMAL)

print(f"✅ saved {len(clean_df)} rows → {OUT_CSV.resolve()}")
