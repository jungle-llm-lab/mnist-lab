# 학습 로그

매일 종료 전에 짧게 기록합니다. 동료 코드나 AI의 구현 제안을 보기 전에는 내가 시도한 내용을 먼저 남기세요.

## Day N - YYYY-MM-DD

### 오늘 직접 구현한 것

- `losses.py`: softmax와 cross-entropy를 묶어서 역전파
- `activations.py`: ReLU 함수의 역전파
- `layers.py`: affine 층의 역전파
- `optimizers.py`: 역전파하여 구한 그라디언트로 파라미터 제자리 갱신
- `traning.py`: forward, loss, backward, update 순으로 연결하여 학습

### 오늘 확인한 shape와 수식

softmax함수와 cross-entropy 함수는 따로 미분하는것보다 묶어서 미분하는 것이 수식이 훨씬 간단하다.

또한 역전파 시 마스크의 기준은 순전파 과정에서 만들어놓은 마스크를 따른다.

### 막힌 지점과 해결 과정

`SoftmaxWithLoss.backward`에서, 각 샘플의 정답 위치에서 1을 빼는 과정에서 자잘한 수정이 있었다.

맨 처음에는 1줄짜리 리스트 컴프리헨션을 사용했으나, 코드가 2줄이 되더라도 간결하게 작성해야겠다 생각했다.

```python
# before
one_hot = np.array([1 if i == j for i in range(num_classes)] for j in self.y_true)

# after
one_hot = np.zeros((batch_size, num_classes))
one_hot[np.arange(batch_size), self.y_true] = 1
```

확실히 고친 후가 더욱 깔끔하게 작성되었다. 더불어서 1일차 때 봤던 문법을 다시한번 짚고 넘어가기를 할 수 있었다.

### 참고 기록

| 출처(교재·동료·AI) | 무엇을 참고했는가 | 내 코드에서 검증한 방법 |
| --- | --- | --- |
| - | - | - |

### 코드 없이 답할 질문 3개 (Day 2)

1. dW, db, dx의 shape는 각각 무엇과 같아야 하는가?
    - 각각 W, b, x의 shape와 같아야 한다.
2. ReLU backward에서 gradient가 0이 되는 위치는 어디인가?
    - ReLU forward에서 값이 음수여서 0으로 바뀐 위치가 역전파 과정에서 그대로 0으로 바뀐다.
3. gradient check는 어떤 두 gradient를 비교하는가?
    - 파라미터 (W, b)중 어느 하나를 아주 작은 값으로 바꿔서 "수치 미분한 gradient"와, 역전파로 넘어온 "해석적 gradient"를 비교해서 오차 범위 안으로 들어오면 성공한다.
    - 정확한지 검사하지 않고 이렇게 검사하는 이유는 cross-entropy 계산 시 log(0) 문제를 회피하려고 1e-7과 같은 값으로 보정했기 때문이다.

### 코드 없이 답할 질문 3개 (Day 3)

4. SGD가 파라미터 배열을 제자리에서 갱신해야 하는 이유는 무엇인가?
    - 배열을 새로 선언하면 모델이 이전 배열만 계속 보게 되고, 손실이 줄어들지 않기 때문이다.
5. Forward, Loss, Backward, Update가 이 순서인 이유는 무엇인가?
    - 순전파 이후 손실을 구한 다음, 이 gradient부터 시작해서 역전파를 해야 파라미터가 옳게 갱신될 수 있기 때문이다.
6. loss 감소가 일반화 성능까지 보장하지 않는 이유는 무엇인가?
    - 여기서 나오는 loss는 학습 데이터에 대한 loss이기 때문에, 일반화시키려고 데이터의 세세한 입력까지 모두 학습시켰다가 일반적인 성능은 되려 낮아질 수 있는 문제가 있다.

