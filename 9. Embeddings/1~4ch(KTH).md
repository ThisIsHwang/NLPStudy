# 워드 임베딩

단어를 AI 처리를 하기 위해서 벡터 형식으로 표현하게 되는데, 대표적인 방식이 전에 배운 원-핫 인코딩이다. 원-핫 벡터는 “강아지"라는 단어를 표현하기 위해 “강아지"를 뜻하는 차원의 값에 1을 주고, 나머지를 0을 주는 방식이다.

이렇듯 단어를 표현하는데에는 여러가지 방식이 존재하고 벡터로 표현하는데에 있어 쓰는 차원에 따라 희소표현 과 밀집 표현으로 구분된다.

### 희소 표현

대표적인 희소 표현은 앞서 말한 원-핫 인코딩이다. 희소표현은 하나의 단어를 표현하기 위해서 단어 하나에 한 차원을 맡고 값을 1로만 표현한다. 나머지는 0으로 표현되어서 차원이 크게 사용되기 때문에 희소 표현이라고 한다.

### 밀집 표현

밀집 표현은 벡터의 차원을 고정시키면 하나의 값이 1이고 나머지 값이 전부 0이 되는 방식이 아닌, 전부 실수가 되어서 차원을 훨씬 줄여서 단어를 표현하는 방식이다.

벡터의 차원이 조밀해졌기 때문에 밀집 벡터라고 한다.

(임베딩 벡터 : 밀집 벡터를 워드 임베딩 과정을 통해 나온 결과)

원-핫 벡터와 임베딩 벡터

차원    |     고차원(단어 집합의 크기), 저차원

다른 표현    |    희소 벡터의 일종, 밀집 벡터의 일종

표현 방법    |    수동, 훈련 데이터로부터 학습함

값의 타입     |     1과0, 실수

## 워드투벡터

단어간 유의미한 유사도를 계산할 수 있는 방식에는 대표적인 방법으로 워드투벡터가 있다.

유사도를 계산할 수 있다는 것은 단어간의 관계가 정의될 수 있다는 것이고, 이는 사칙연산과 같은 연산을 할 수 있다는 사실로 이어진다.

대표적인 계산의 예시는

**한국 - 서울 + 도쿄 = 일본**

**박찬호 - 야구 + 축구 = 호나우두**

가 있다.

### 분산 표현

분산 표현 방법은 기본적으로 분포 가설이라는 가정 하에 만들어진 표현 방법이다. 이 가정은 **'비슷한 문맥에서 등장하는 단어들은 비슷한 의미를 가진다'** 라는 가정이다.

이런식으로 표현된 단어들은 “의미"라는 특징이 부여되기 때문에 다차원에 하나의 값에만 1을 쓰고 나머지는 0을 쓸 필요 없이 값을 분산시켜 하나의 차원이 “단어"가 아닌 “의미"를 넣어 실수가 대입되고, 각 차원의 값을 통해서 서로의 관계를 알아낼 수 있다.

워드투벡터의 학습 방식에는 CBOW(Continuous Bag of Words)와 Skip-Gram 두 가지 방식이 있다. CBOW는 주변에 있는 단어들을 입력으로 중간에 있는 단어들을 예측하는 방법이고 , Skip-Gram은 반대로 중간에 있는 단어로 주변 단어들을 예측하는 방법이다.

## 워드투벡터 실습

### 영어 Word2Vec 만들기

**패키지  import**

```python
import re
import urllib.request
import zipfile
from lxml import etree
from nltk.tokenize import word_tokenize, sent_tokenize
```

**훈련 데이터 다운로드**

```python
urllib.request.urlretrieve("https://raw.githubusercontent.com/ukairia777/tensorflow-nlp-tutorial/main/09.%20Word%20Embedding/dataset/ted_en-20160408.xml", filename="ted_en-20160408.xml")
```

**훈련 데이터 전처리하기**

```python
targetXML = open('ted_en-20160408.xml', 'r', encoding='UTF8')
target_text = etree.parse(targetXML)

# xml 파일로부터 <content>와 </content> 사이의 내용만 가져온다.
parse_text = '\n'.join(target_text.xpath('//content/text()'))

# 정규 표현식의 sub 모듈을 통해 content 중간에 등장하는 (Audio), (Laughter) 등의 배경음 부분을 제거.
# 해당 코드는 괄호로 구성된 내용을 제거.
content_text = re.sub(r'\([^)]*\)', '', parse_text)

# 입력 코퍼스에 대해서 NLTK를 이용하여 문장 토큰화를 수행.
sent_text = sent_tokenize(content_text)

# 각 문장에 대해서 구두점을 제거하고, 대문자를 소문자로 변환.
normalized_text = []
for string in sent_text:
     tokens = re.sub(r"[^a-z0-9]+", " ", string.lower())
     normalized_text.append(tokens)

# 각 문장에 대해서 NLTK를 이용하여 단어 토큰화를 수행.
result = [word_tokenize(sentence) for sentence in normalized_text]
print('총 샘플의 개수 : {}'.format(len(result)))
```

```python
총 샘플의 개수 : 273424
```

**Word2Vec 훈련시키기**

```python
from gensim.models import Word2Vec
from gensim.models import KeyedVectors

model = Word2Vec(sentences=result, size=100, window=5, min_count=5, workers=4, sg=0)
```

파라미터 값의 의미

- size = 워드 벡터의 특징 값, 임베딩 된 벡터의 차원
- window = 컨텍스트 윈도우 크기
- min_count = 단어 최소 빈도 수 제한 (해당 빈도 보다 적은 단어들은 학습하지 않는다.)
- workers = 학습을 위한 프로세스 수
- sg = 0은 CBOW 방식, 1은 Skip-gram 방식

이렇게 하면 학습이 진행이 완료된다. 학습이 진행된 결과를 눈으로 확인해 보기 위해 Word2Vec은 model.wv.most_similar을 지원한다. 해당 함수 를 통해서 인자로 단어를 넣어주면 유사한 단어들을 확인할 수 있다.

```python
model_result = model.wv.most_similar("man")
print(model_result)
```

```python
[('woman', 0.842622697353363), ('guy', 0.8178728818893433),
('boy', 0.7774451375007629), ('lady', 0.7767927646636963),
('girl', 0.7583760023117065), ('gentleman', 0.7437191009521484),
('soldier', 0.7413754463195801), ('poet', 0.7060446739196777),
('kid', 0.6925194263458252), ('friend', 0.6572611331939697)]
```

## 네거티브 샘플링을 이용한 Word2Vec 구현

### 네거티브 샘플링

네거티브 샘플링은 Word2Vec이 학습 과정에서 전체 단어 집합이 아니라 일부 단어 집합에만 집중할 수 있도록 하는 방법이다.

이는 기존의 단어 집합의 크기만큼의 선택지를 두고 다중 클래스 분류 문제를 풀던 Word2Vec보다 훨씬 연산량에서 효율적이다.

**패키지 import하기**

```python
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from sklearn.datasets import fetch_20newsgroups
from tensorflow.keras.preprocessing.text import Tokenizer
```

**이번 실습에서는 주변 단어의 관계가 성립해야 학습이 가능하기 때문에 전처리 과정에서 이를 만족하지 않는 샘플들을 제거해줘야 한다.**

```python
dataset = fetch_20newsgroups(shuffle=True, random_state=1, remove=('headers', 'footers', 'quotes'))
documents = dataset.data
print('총 샘플 수 :',len(documents))
```

```python
총 샘플 수 : 11314
```

**전처리 진행 ( 불필요한 토큰 제거, 소문자화를 통한 정규화, NULL값 제거)**

```python
news_df = pd.DataFrame({'document':documents})
# 특수 문자 제거
news_df['clean_doc'] = news_df['document'].str.replace("[^a-zA-Z]", " ")
# 길이가 3이하인 단어는 제거 (길이가 짧은 단어 제거)
news_df['clean_doc'] = news_df['clean_doc'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))
# 전체 단어에 대한 소문자 변환
news_df['clean_doc'] = news_df['clean_doc'].apply(lambda x: x.lower())
```

```python
news_df.dropna(inplace=True)
print('총 샘플 수 :',len(news_df))
```

```python
# 불용어를 제거
stop_words = stopwords.words('english')
tokenized_doc = news_df['clean_doc'].apply(lambda x: x.split())
tokenized_doc = tokenized_doc.apply(lambda x: [item for item in x if item not in stop_words])
tokenized_doc = tokenized_doc.to_list()
```

```python
#정수 인코딩
tokenizer = Tokenizer()
tokenizer.fit_on_texts(tokenized_doc)

word2idx = tokenizer.word_index
idx2word = {value : key for key, value in word2idx.items()}
encoded = tokenizer.texts_to_sequences(tokenized_doc)
```

**네거티브 샘플링을 통한 데이터셋 구성하기**

네거티브 샘플링을 위해서 케라스에서 제공하는 전처리 도구인 skipgrams를 사용했다. 어떤 전처리가 수행된는지 결과를 학인하기 위해서 상위 10개의 그룹 샘플만 사용.

```python
from tensorflow.keras.preprocessing.sequence import skipgrams
# 네거티브 샘플링
skip_grams = [skipgrams(sample, vocabulary_size=vocab_size, window_size=10) for sample in encoded[:10]]
```

```python
# 첫번째 샘플인 skip_grams[0] 내 skipgrams로 형성된 데이터셋 확인
pairs, labels = skip_grams[0][0], skip_grams[0][1]
for i in range(5):
    print("({:s} ({:d}), {:s} ({:d})) -> {:d}".format(
          idx2word[pairs[i][0]], pairs[i][0], 
          idx2word[pairs[i][1]], pairs[i][1], 
          labels[i]))
```

```python
(commited (7837), badar (34572)) -> 0
(whole (217), realize (1036)) -> 1
(reason (149), commited (7837)) -> 1
(letter (705), rediculous (15227)) -> 1
(reputation (5533), midonrnax (47527)) -> 0
```

**뉴스 그룹 샘플이 가지고 있는 pair, label 개수 확인**

```python
# 첫번째 뉴스그룹 샘플에 대해서 생긴 pairs와 labels의 개수
print(len(pairs))
print(len(labels))
```

```python
2220
2220
```

**모든 뉴스 그룹 샘플에 대해서 실행**

```python
skip_grams = [skipgrams(sample, vocabulary_size=vocab_size, window_size=10) for sample in encoded]
```

**Skip-Gram with Negative Sampling 구현하기**

```python
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Embedding, Reshape, Activation, Input
from tensorflow.keras.layers import Dot
from tensorflow.keras.utils import plot_model
from IPython.display import SVG
```

**임베딩 벡터 차원을 100으로 설정 & 두 개의 임베딩 층 추가.**

```python
embedding_dim = 100

# 중심 단어를 위한 임베딩 테이블
w_inputs = Input(shape=(1, ), dtype='int32')
word_embedding = Embedding(vocab_size, embedding_dim)(w_inputs)

# 주변 단어를 위한 임베딩 테이블
c_inputs = Input(shape=(1, ), dtype='int32')
context_embedding  = Embedding(vocab_size, embedding_dim)(c_inputs)
```

**각 임베딩 테이블은 중심 단어와 주변 단어 각각을 위한 임베딩 테이블이며 각 단어는 임베딩 테이블을 거쳐서 내적을 수행하고, 내적의 결과는 1 또는 0을 예측하기 위해서 시그모이드 함수를 활성화 함수로 거쳐 최종 예측값 얻기**

```python
dot_product = Dot(axes=2)([word_embedding, context_embedding])
dot_product = Reshape((1,), input_shape=(1, 1))(dot_product)
output = Activation('sigmoid')(dot_product)

model = Model(inputs=[w_inputs, c_inputs], outputs=output)
model.summary()
model.compile(loss='binary_crossentropy', optimizer='adam')
plot_model(model, to_file='model3.png', show_shapes=True, show_layer_names=True, rankdir='TB')
```

**5에포크 수행**

```python
for epoch in range(1, 6):
    loss = 0
    for _, elem in enumerate(skip_grams):
        first_elem = np.array(list(zip(*elem[0]))[0], dtype='int32')
        second_elem = np.array(list(zip(*elem[0]))[1], dtype='int32')
        labels = np.array(elem[1], dtype='int32')
        X = [first_elem, second_elem]
        Y = labels
        loss += model.train_on_batch(X,Y)  
    print('Epoch :',epoch, 'Loss :',loss)
```

```python
Epoch: 1 Loss: 4339.997158139944
Epoch: 2 Loss: 3549.69356325455
Epoch: 3 Loss: 3295.072506020777
Epoch: 4 Loss: 3038.1063768607564
Epoch: 5 Loss: 2790.9479411702487
```

**결과 확인**

```python
import gensim

f = open('vectors.txt' ,'w')
f.write('{} {}\n'.format(vocab_size-1, embed_size))
vectors = model.get_weights()[0]
for word, i in tokenizer.word_index.items():
    f.write('{} {}\n'.format(word, ' '.join(map(str, list(vectors[i, :])))))
f.close()

# 모델 로드
w2v = gensim.models.KeyedVectors.load_word2vec_format('./vectors.txt', binary=False)
```

```python
w2v.most_similar(positive=['soldiers'])
```

```python
[('lebanese', 0.7539176940917969),
 ('troops', 0.7515299916267395),
 ('occupying', 0.7322258949279785),
 ('attacking', 0.7247686386108398),
 ('villagers', 0.7217503786087036),
 ('israeli', 0.7071422338485718),
 ('villages', 0.7000206708908081),
 ('wounded', 0.6976917386054993),
 ('lebanon', 0.6933401823043823),
 ('arab', 0.692956268787384)]
```

```python
w2v.most_similar(positive=['doctor'])
```

```python
[('nerve', 0.6576169729232788),
 ('migraine', 0.6502577066421509),
 ('patient', 0.6377599835395813),
 ('disease', 0.6300654411315918),
 ('quack', 0.6101700663566589),
 ('cardiac', 0.606243371963501),
 ('infection', 0.6030253171920776),
 ('medication', 0.6001783013343811),
 ('suffering', 0.593578040599823),
 ('hurt', 0.5818471908569336)]
```