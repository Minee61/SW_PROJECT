# -*- coding: utf-8 -*-
"""
wiki_style_kg.py — 한국어 '문체·표기·맞춤법' 위키 페이지 → 삼요소 트리플 CSV

🔹 개선 포인트
  • 카테고리 비어 있으면 예비 페이지 30 개 사용
  • 문장 길이 필터 10–200 자로 완화, 제목 포함 조건 제거
  • predicate 3 종: defines / example_of / contrast_with
  • object  : Okt 형태소 + TF-IDF 상위 3 명사 키워드
  • 중복 트리플 제거
  • triples.csv + README_LICENSE.txt 자동 생성

🔹 의존
  pip install "wikipedia-api<0.6" scikit-learn pandas tqdm konlpy==0.6.0
  sudo apt-get install -y default-jdk   # (Ubuntu, Konlpy JPype 용)
"""

import re, csv, pathlib
from typing import List, Tuple
import wikipediaapi, tqdm
from konlpy.tag import Okt
from sklearn.feature_extraction.text import TfidfVectorizer

# ──────────────────────────────────
# 0. 설정
# ──────────────────────────────────
USER_AGENT = "AdvWikiKG/0.3 (https://example.com)"
WIKI = wikipediaapi.Wikipedia("ko", headers={"User-Agent": USER_AGENT})

CATEGORY  = "한국어 문체"     # 카테고리 먼저 시도
MAX_PAGES = 30

LEN_MIN, LEN_MAX = 10, 200    # ← 길이 조건 완화
TOKEN_RE = r"[가-힣A-Za-z]{2,}"

SENT_SPLIT = re.compile(r"[.!?]\s|\n")
okt = Okt()

PRED_PATTERNS = [
    (re.compile(r"예를 들어"), "example_of"),
    (re.compile(r"[와과] 다르게"), "contrast_with"),
]
DEFAULT_PRED = "defines"

FALLBACK_PAGES = [
    # 기본 10개 + 추가 20개 (존재 여부는 위키 검색으로 확인)
    "구어체", "문어체", "높임법", "경어법", "표준어", "방언",
    "인터넷 속어", "금기어", "국어 문장부호", "맞춤법",
    "존댓말", "반말", "격식체", "속어", "신조어",
    "관용구", "비속어", "외래어", "띄어쓰기", "두음 법칙",
    "사투리", "의성어", "의태어", "종결어미", "의문문",
    "부정문", "피동", "사동", "중의적 표현", "로마자 표기법"
]

# ──────────────────────────────────
# 1. 함수: 카테고리에서 페이지 수집
# ──────────────────────────────────
def collect_pages(category: str, limit: int) -> List[Tuple[str, str]]:
    cat = WIKI.page(f"Category:{category}")
    pages = []
    for title, page in cat.categorymembers.items():
        if page.ns == 0:
            pages.append((title, page.fullurl))
        if len(pages) >= limit:
            break
    return pages

# ──────────────────────────────────
# 2. 페이지 목록 결정
# ──────────────────────────────────
pages = collect_pages(CATEGORY, MAX_PAGES)

if not pages:
    print("⚠️  카테고리 비어 있음 → 예비 페이지 사용")
    pages = [
        (title, WIKI.page(title).fullurl)
        for title in FALLBACK_PAGES
        if WIKI.page(title).exists()
    ]

if not pages:
    raise SystemExit("❌  예비 페이지도 모두 실패 — 제목을 확인하세요.")

# ──────────────────────────────────
# 3. 문장 수집
# ──────────────────────────────────
sentences, meta = [], []   # sentence, (subject, url)
for subj, url in tqdm.tqdm(pages, desc="fetch pages"):
    page = WIKI.page(subj)
    if not page.exists():
        continue
    for sent in SENT_SPLIT.split(page.text):
        sent = sent.strip()
        if LEN_MIN < len(sent) < LEN_MAX:
            sentences.append(sent)
            meta.append((subj, url))

if len(sentences) < 300:
    raise SystemExit(f"❗  수집 문장 수가 {len(sentences)}개 — 더 늘려야 합니다.")

# ──────────────────────────────────
# 4. TF-IDF 키워드 추출
# ──────────────────────────────────
vectorizer = TfidfVectorizer(max_features=8000, token_pattern=TOKEN_RE)
X = vectorizer.fit_transform(sentences)
idx2term = {i: t for t, i in vectorizer.vocabulary_.items()}

triples, seen = [], set()
for i, sent in enumerate(sentences):
    subj, url = meta[i]

    # predicate 선택
    pred = next((tag for pat, tag in PRED_PATTERNS if pat.search(sent)), DEFAULT_PRED)

    # TF-IDF 상위 3 명사
    row = X.getrow(i)
    if row.nnz == 0:
        continue
    top_idx = row.indices[row.data.argsort()[::-1][:3]]
    obj = " ".join(idx2term[j] for j in top_idx)
    if not obj:
        continue

    key = (subj, pred, obj)
    if key in seen:
        continue
    seen.add(key)
    triples.append([subj, pred, obj, url])

# ──────────────────────────────────
# 5. CSV + 라이선스 파일 저장
# ──────────────────────────────────
out_csv = pathlib.Path("triples.csv")
with out_csv.open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["subject", "predicate", "object", "src_url"])
    writer.writerows(triples)

readme = pathlib.Path("README_LICENSE.txt")
readme.write_text(
    "Data Source\n-----------\n"
    "본 트리플은 한국어 위키백과 덤프(CC BY-SA 3.0)에서 자동 추출된 fact 정보입니다.\n"
    "원문 문장은 포함되지 않았으며, 재배포 시 CC BY-SA 3.0 출처 고지가 필요합니다.\n",
    encoding="utf-8",
)

print(f"✓ {len(triples)} triples saved → {out_csv.resolve()}")
