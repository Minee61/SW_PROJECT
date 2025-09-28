# SW_PROJECT

## 개요  
이 프로젝트는 **RAG 기반 지식 검색 + 자연어 질의 응답 기능**을 구현한 소프트웨어 프로젝트입니다.

## 주요 기능  
- **KG 기반 질의 응답**  
  `RAP_final_kg_pipeline.py` 등 파이프라인 코드를 통해 지식 그래프(KG)를 구축하고, 질의에 대해 적절한 응답을 생성  
- **데이터 처리 / 전처리**  
  `kg.csv`, `test.csv`, `train.csv` 등을 사용하여 데이터셋을 핸들링하고, 모델 학습 및 평가를 위한 전처리 수행  
- **테스트 및 평가**  
  `kg_performance_eval(1).ipynb` 파일을 통해 성능 평가 지표를 시각화 및 분석  
- **요구사항 / 패키지 관리**  
  `req.txt` 파일에 필요한 Python 라이브러리 명시 (예: transformers, torch 등)  

## 사용 도구  
- **Python**: 핵심 언어  
- **Jupyter Notebook**: 데이터 탐색 및 평가 분석을 위해 사용  
- **라이브러리**: Pandas, NumPy, 기타 머신러닝/딥러닝 라이브러리 (req.txt 참고) 
