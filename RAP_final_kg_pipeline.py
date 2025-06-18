import pandas as pd
import random
from datasets import Dataset
from transformers import AutoTokenizer

#  Load Knowledge Graph CSV
kg_df = pd.read_csv("RAP_Study/kg.csv") 
print("원본 KG 크기:", kg_df.shape)

# Rule-based Filtering
allowed_relations = ["설립자", "소속", "위치", "개발자", "출시일"]
kg_df = kg_df[kg_df["relation"].isin(allowed_relations)]
kg_df = kg_df[kg_df["subject"].str.len() > 1]
kg_df = kg_df[kg_df["object"].str.len() > 1]
kg_df = kg_df[kg_df["subject"].str.isalnum()]
kg_df = kg_df[kg_df["object"].str.isalnum()]
print("필터링된 KG 크기:", kg_df.shape)

# 자연어 문장으로 변환
kg_df["kg_sentence"] = kg_df.apply(lambda row: f"{row['subject']}는 {row['relation']}가 {row['object']}입니다.", axis=1)
kg_sentences = kg_df["kg_sentence"].tolist()

train = pd.read_csv("RAP_Study/train.csv")
test = pd.read_csv("RAP_Study/test.csv")

# sentence1~4 이어 붙이기
def combine_sentences(row):
    return f"{row['sentence1']} {row['sentence2']} {row['sentence3']} {row['sentence4']}"

# KG 문장 랜덤 삽입
def add_kg_hint(row):
    main_text = combine_sentences(row)
    if kg_sentences:
        kg_hint = random.choice(kg_sentences)
        return main_text + " [KG] " + kg_hint
    else:
        return main_text  

train["text"] = train.apply(add_kg_hint, axis=1)
test["text"] = test.apply(lambda row: combine_sentences(row), axis=1)

#Tokenization (KLUE-RoBERTa)
tokenizer = AutoTokenizer.from_pretrained("klue/roberta-base")
train_dataset = Dataset.from_pandas(train[["text", "label"]])
test_dataset = Dataset.from_pandas(test[["text"]])

train_tokenized = train_dataset.map(lambda x: tokenizer(x["text"], truncation=True, padding="max_length"), batched=True)
test_tokenized = test_dataset.map(lambda x: tokenizer(x["text"], truncation=True, padding="max_length"), batched=True)

print("Tokenization 완료")


#------------------
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

# 모델 로드
model = AutoModelForSequenceClassification.from_pretrained("klue/roberta-base", num_labels=train["label"].nunique())

# 평가 함수 정의
def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='weighted')
    return {"accuracy": acc, "f1": f1}

# 학습 설정
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="f1"
)

# Trainer 생성
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset=train_tokenized,  # 현재 validation 분리가 없어서 train으로 평가
    compute_metrics=compute_metrics,
)

# 학습 실행
trainer.train()

# 최종 평가 결과 출력
metrics = trainer.evaluate()
print("최종 평가 결과:")
print(metrics)
