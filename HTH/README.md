# ****딥 러닝의 학습 방법****

손실 함수, 옵티마이저, 에포크

**손실 함수:** 실제값과 예측값의 차이

**손실함수 종류**

****1) MSE(Mean Squared Error, MSE)****

****2) 이진 크로스 엔트로피(Binary Cross-Entropy)****

****3) 카테고리칼 크로스 엔트로피(Categorical Cross-Entropy)****

****4) 그 외에 다양한 손실 함수들****

https://www.tensorflow.org/api_docs/python/tf/keras/losses

**배치 크기 경사하강법**

배치: 가중치 등의 매개 변수의 값을 조정하기 위해 사용하는 데이터의 양

****1) 배치 경사 하강법(Batch Gradient Descent)****

배치 경사 하강법은 옵티마이저 중 하나로 오차(loss)를 구할 때 전체 데이터를 고려

딥 러닝에서는 전체 데이터에 대한 한 번의 훈련 횟수를 1 에포크

배치 경사 하강법은 전체 데이터를 고려해서 학습하므로 한 번의 매개 변수 업데이트에 시간이 오래 걸리며, 메모리를 크게 요구한다는 단점이 있습니다.

****2) 배치 크기가 1인 확률적 경사 하강법(Stochastic Gradient Descent, SGD)****

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/b8502f8d-8040-45d7-ac10-80a0bdb43b80/Untitled.png)

배치 크기가 1인 확률적 경사 하강법은 매개변수 값을 조정 시 전체 데이터가 아니라 랜덤으로 선택한 하나의 데이터에 대해서만 계산하는 방법입니다.

****3) 미니 배치 경사 하강법(Mini-Batch Gradient Descent)****

전체 데이터도, 1개의 데이터도 아닐 때, 배치 크기를 지정하여 해당 데이터 개수만큼에 대해서 계산하여 매개 변수의 값을 조정하는 경사 하강법을 미니 배치 경사 하강법이라고 합니다.

****3. 옵티마이저(Optimizer)****

****1) 모멘텀(Momentum)****

경사하강법 + 관성 

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/91a5083e-1756-4cec-ba39-69993b9c1929/Untitled.png)

전체 함수에 걸쳐 최소값을 **글로벌 미니멈(Global Minimum)**
 이라고 하고, 글로벌 미니멈이 아닌 특정 구역에서의 최소값인 **로컬 미니멈(Local Minimum)**
 이라고 합니다.

**2) 아다그라드(Adagrad)**

아다그라드는 각 매개변수에 서로 다른 학습률을 적용시킵니다. 이때 변화가 많은 매개변수는 학습률이 작게 설정되고 변화가 적은 매개변수는 학습률을 높게 설정시킵니다.

****3) 알엠에스프롭(RMSprop)****

나중에 가서는 학습률이 지나치게 떨어진다는 단점이 있는데 이를 다른 수식으로 대체하여 이러한 단점을 개선하였습니다.

****4) 아담(Adam)****

아담은 알엠에스프롭과 모멘텀 두 가지를 합친 듯한 방법으로, 방향과 학습률 두 가지를 모두 잡기 위한 방법입니다.

```python
adam = tf.keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['acc'])
```

****5. 에포크와 배치 크기와 이터레이션(Epochs and Batch size and Iteration)****

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/caa3d973-3b77-4f65-9616-58c4d14c3170/Untitled.png)

****1) 에포크(Epoch)****

에포크란 인공 신경망에서 전체 데이터에 대해서 순전파와 역전파가 끝난 상태를 말합니다. 이 에포크 횟수가 지나치거나 너무 적으면 앞서 배운 과적합과 과소적합이 발생할 수 있습니다.

****2) 배치 크기(Batch size)****

배치 크기는 몇 개의 데이터 단위로 매개변수를 업데이트 하는지를 말합니다. 현실에 비유하면 문제지에서 몇 개씩 문제를 풀고나서 정답지를 확인하느냐의 문제입니다.

****3) 이터레이션(Iteration) 또는 스텝(Step)****

이터레이션이란 한 번의 에포크를 끝내기 위해서 필요한 배치의 수를 말합니다.

전체 데이터  / 배치크기 == 이터레이션 수

****04-3) 역전파(BackPropagation) 이해하기****

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/ef372383-6bb5-4320-93b5-893849eef91d/Untitled.png)

> 이후 시그모이드 통과한 것
> 

****2. 순전파(Forward Propagation)****

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/4ffad1fe-f0f2-46d2-a5f8-0e201de63b8f/Untitled.png)

****3. 역전파**** 

[https://wikidocs.net/37406](https://wikidocs.net/37406)

****05) 과적합(Overfitting)을 막는 방법들****

모델이 과적합되면 훈련 데이터에 대한 정확도는 높을지라도, 새로운 데이터. 즉, 검증 데이터나 테스트 데이터에 대해서는 제대로 동작하지 않습니다.

****1. 데이터의 양을 늘리기****

****2. 모델의 복잡도 줄이기****

****3. 가중치 규제(Regularization) 적용하기****

- L1 규제 : 가중치 w들의 절대값 합계를 비용 함수에 추가합니다. L1 노름이라고도 합니다.
- L2 규제 : 모든 가중치 w들의 제곱합을 비용 함수에 추가합니다. L2 노름이라고도 합니다.

L2 규제가 더 잘 동작하므로 L2 규제를 사용 권장

****4. 드롭아웃(Dropout)****

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/8eae1a8e-ef07-4b8e-a273-96b2105bf9fa/Untitled.png)

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/4baf5753-cfa3-43e0-9fe4-82986800d520/Untitled.png)

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Dense

max_words = 10000
num_classes = 46

model = Sequential()
model.add(Dense(256, input_shape=(max_words,), activation='relu'))
model.add(Dropout(0.5)) # 드롭아웃 추가. 비율은 50%
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5)) # 드롭아웃 추가. 비율은 50%
model.add(Dense(num_classes, activation='softmax'))
```

****06) 기울기 소실(Gradient Vanishing)과 폭주(Exploding)****

입력층에 가까운 층들에서 가중치들이 업데이트가 제대로 되지 않으면 결국 최적의 모델을 찾을 수 없게 됩니다. 이를 **기울기 소실(Gradient Vanishing)**

기울기가 점차 커지더니 가중치들이 비정상적으로 큰 값이 되면서 결국 발산되기도 합니다. 이를 **기울기 폭주(Gradient Exploding)**

****1. ReLU와 ReLU의 변형들****

시그모이드 함수를 사용하면 입력의 절대값이 클 경우에 시그모이드 함수의 출력값이 0 또는 1에 수렴하면서 기울기가 0에 가까워집니다

은닉층의 활성화 함수로 시그모이드나 하이퍼볼릭탄젠트 함수 대신에 ReLU나 ReLU의 변형 함수와 같은 Leaky ReLU를 사용하는 것입니다.

- 은닉층에서는 시그모이드 함수를 사용하지 마세요.
- Leaky ReLU를 사용하면 모든 입력값에 대해서 기울기가 0에 수렴하지 않아 죽은 ReLU 문제를 해결합니다.
- 은닉층에서는 ReLU나 Leaky ReLU와 같은 ReLU 함수의 변형들을 사용하세요.

****2. 그래디언트 클리핑(Gradient Clipping)****

그래디언트 클리핑은 말 그대로 기울기 값을 자르는 것을 의미합니다. 기울기 폭주를 막기 위해 임계값을 넘지 않도록 값을 자릅니다.

## **3. 가중치 초기화(Weight initialization)**

 다시 말해 가중치 초기화만 적절히 해줘도 기울기 소실 문제과 같은 문제를 완화시킬 수 있습니다.

****1) 세이비어 초기화(Xavier Initialization)****

정규분포, 균등 분포

****2) He 초기화(He initialization)****

****4. 배치 정규화(Batch Normalization)****

기울기 소실이나 폭주를 예방하는 또 다른 방법은 배치 정규화(Batch Normalization)입니다.

****1) 내부 공변량 변화(Internal Covariate Shift)****

층 별로 입력 데이터 분포가 달라지는 현상

- 공변량 변화는 훈련 데이터의 분포와 테스트 데이터의 분포가 다른 경우를 의미합니다.
- 내부 공변량 변화는 신경망 층 사이에서 발생하는 입력 데이터의 분포 변화를 의미합니다.
- 

****2) 배치 정규화(Batch Normalization)****

배치 정규화(Batch Normalization)는 표현 그대로 한 번에 들어오는 배치 단위로 정규화하는 것을 말합니다.

입력에 대해 평균을 0으로 만들고, 정규화를 합니다. 그리고 정규화 된 데이터에 대해서 스케일과 시프트를 수행합니다.

- 배치 정규화를 사용하면 시그모이드 함수나 하이퍼볼릭탄젠트 함수를 사용하더라도 기울기 소실 문제가 크게 개선됩니다.
- 가중치 초기화에 훨씬 덜 민감해집니다.
- 훨씬 큰 학습률을 사용할 수 있어 학습 속도를 개선시킵니다.
- 미니 배치마다 평균과 표준편차를 계산하여 사용하므로 훈련 데이터에 일종의 잡음 주입의 부수 효과로 과적합을 방지하는 효과도 냅니다. 다시 말해, 마치 드롭아웃과 비슷한 효과를 냅니다. 물론, 드롭 아웃과 함께 사용하는 것이 좋습니다.
- 배치 정규화는 모델을 복잡하게 하며, 추가 계산을 하는 것이므로 테스트 데이터에 대한 예측 시에 실행 시간이 느려집니다. 그래서 서비스 속도를 고려하는 관점에서는 배치 정규화가 꼭 필요한지 고민이 필요합니다.
- 배치 정규화의 효과는 굉장하지만 내부 공변량 변화때문은 아니라는 논문도 있습니다. :

****5. 층 정규화(Layer Normalization)****

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/af9a226e-06d6-48cc-8a88-671a0f93b68f/Untitled.png)

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/27e0dd27-6746-4c74-9825-d55be2a3e4e0/Untitled.png)

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/9dbb6ec2-5866-4ba4-8c15-e40f6550d69a/Untitled.png)

동일한 특성 개수를 가진 다수의 샘플
