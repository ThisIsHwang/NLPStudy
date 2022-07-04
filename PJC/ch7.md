Keras를 통한 NLP 개요
---
1. 전처리(Preprocessing)
- Tokenizer()
    - Text를 token으로 변환
- pad_sequence()
    - 훈련 데이터 간 길이를 통일
    - 짧을 경우 앞 혹은 뒤를 임의의 값(ex. 0)으로 채움
    - 길이가 길 경우 앞 혹은 뒷부분을 자름
2. 워드 임베딩(Word Embedding)
- Embedding()
    - token 단위의 단어를 vector로 변환
3. 모델링(Modeling)
- Sequential()
    - 모델의 layer를 쌓는 형태로 구현
    - add()를 통해 각 층을 순서대로 쌓음
- Dense()
    - fully-connected layer를 생성
- summary()
    - model의 정보를 요약해서 보여줌
4. 컴파일과 훈련(Compile and Training)
- compile()
    - optimizer, loss function, metric 설정
- fit()
    - 입출력 데이터, epoch, batch_size 설정 후 train
    - validation_data 설정 가능
5. 평가와 예측(Evaluation and Prediction)
- evaluate()
    - test dataset을 통한 모델 평가
- predict()
    - 입력에 따른 모델의 결과값 출력
6. 모델 저장과 로드(Model Save and Load)
- save()
    - 생성한 모델 저장
    - 모델 구조, weight, bias 등
- load_model()
    - 저장된 모델 불러오기


Keras의 Model 생성 API들
---
1. Sequential API
- layer를 순서대로 쌓을 때 사용
- 가장 기본적인 방식
2. Functional API
- layer를 병렬로 쌓을 수 있음
- 보다 복잡한 방식
3. Subclassing API
- keras 라이브러리에 있는 basic layer들의 변형이 필요할 때 사용
- ex) layer 혹은 model의 output을 call method를 통해 변형 가능
