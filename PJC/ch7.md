Keras를 통한 NLP 개요
---
1. 전처리(Preprocessing)
- Tokenizer()
- pad_sequence()
2. 워드 임베딩(Word Embedding)
- Embedding()
3. 모델링(Modeling)
- Sequential()
- Dense()
- summary()
4. 컴파일과 훈련(Compile and Training)
- compile()
- fit()
5. 평가와 예측(Evaluation and Prediction)
- evaluate()
- predict()
6. 모델 저장과 로드(Model Save and Load)
- save()
- load_model()


Keras의 Model 생성 API들
---
1. Sequential API
- layer를 직렬로 쌓을 때
- 가장 기초적인 방식
2. Functional API
- layer를 병렬로 쌓을 수 있음
- 보다 보잡한 방식
3. Subclassing API
- keras 라이브러리에 있는 basic layer들의 변형이 필요할 때 사용
- ex) layer 혹은 model의 output을 call method를 통해 변형 가능
