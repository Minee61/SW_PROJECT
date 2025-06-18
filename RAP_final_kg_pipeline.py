
import pandas as pd
import random
from datasets import Dataset
from transformers import AutoTokenizer

# 1. Load Knowledge Graph CSV
kg_df = pd.read_csv("kg.csv")  # 필수: subject, relation, object 컬럼 포함
print("원본 KG 크기:", kg_df.shape)

# 2. Rule-based Filtering
allowed_relations = ["설립자", "소속", "위치", "개발자", "출시일"]
kg_df = kg_df[kg_df["relation"].isin(allowed_relations)]
kg_df = kg_df[kg_df["subject"].str.len() > 1]
kg_df = kg_df[kg_df["object"].str.len() > 1]
kg_df = kg_df[kg_df["subject"].str.isalnum()]
kg_df = kg_df[kg_df["object"].str.isalnum()]
print("필터링된 KG 크기:", kg_df.shape)

# 3. KG triple → 자연어 문장으로 변환
kg_df["kg_sentence"] = kg_df.apply(lambda row: f"{row['subject']}는 {row['relation']}가 {row['object']}입니다.", axis=1)
kg_sentences = kg_df["kg_sentence"].tolist()

# 4. Load train/test data
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# 5. sentence1~4 이어 붙이기
def combine_sentences(row):
    return f"{row['sentence1']} {row['sentence2']} {row['sentence3']} {row['sentence4']}"

# 6. KG 문장 랜덤 삽입
def add_kg_hint(row):
    main_text = combine_sentences(row)
    if kg_sentences:
        kg_hint = random.choice(kg_sentences)
        return main_text + " [KG] " + kg_hint
    else:
        return main_text  # KG 없으면 원본만

train["text"] = train.apply(add_kg_hint, axis=1)
test["text"] = test.apply(lambda row: combine_sentences(row), axis=1)

# 7. Tokenization (KLUE-RoBERTa)
tokenizer = AutoTokenizer.from_pretrained("klue/roberta-base")
train_dataset = Dataset.from_pandas(train[["text", "label"]])
test_dataset = Dataset.from_pandas(test[["text"]])

train_tokenized = train_dataset.map(lambda x: tokenizer(x["text"], truncation=True, padding="max_length"), batched=True)
test_tokenized = test_dataset.map(lambda x: tokenizer(x["text"], truncation=True, padding="max_length"), batched=True)

print("Tokenization 완료")
