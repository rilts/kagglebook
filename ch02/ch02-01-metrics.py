import numpy as np
import pandas as pd

# -----------------------------------
# 회귀
# -----------------------------------
# rmse

from sklearn.metrics import mean_squared_error

# y_true가 실제값, y_pred가 예측값
y_true = [1.0, 1.5, 2.0, 1.2, 1.8]
y_pred = [0.8, 1.5, 1.8, 1.3, 3.0]

rmse = np.sqrt(mean_squared_error(y_true, y_pred))
print(rmse)
# 0.5532

# -----------------------------------
# 2진분류
# -----------------------------------
# 오차행렬

from sklearn.metrics import confusion_matrix

# 0, 1로 표현되는 2진분류의 실제값과 예측값
y_true = [1, 0, 1, 1, 0, 1, 1, 0]
y_pred = [0, 0, 1, 1, 0, 0, 1, 1]

tp = np.sum((np.array(y_true) == 1) & (np.array(y_pred) == 1))
tn = np.sum((np.array(y_true) == 0) & (np.array(y_pred) == 0))
fp = np.sum((np.array(y_true) == 0) & (np.array(y_pred) == 1))
fn = np.sum((np.array(y_true) == 1) & (np.array(y_pred) == 0))

confusion_matrix1 = np.array([[tp, fp],
                              [fn, tn]])
print(confusion_matrix1)
# array([[3, 1],
#        [2, 2]])

# scikit-learn의 metrics모듈의 confusion_matrix이라도 작성할 수 있지만 오차행렬의 요소 배치가 다르기에 주의 필요
confusion_matrix2 = confusion_matrix(y_true, y_pred)
print(confusion_matrix2)
# array([[2, 1],
#        [2, 3]])

# -----------------------------------
# accuracy

from sklearn.metrics import accuracy_score

# 0, 1로 표현되는 2진분류의 실제값과 예측값
y_true = [1, 0, 1, 1, 0, 1, 1, 0]
y_pred = [0, 0, 1, 1, 0, 0, 1, 1]
accuracy = accuracy_score(y_true, y_pred)
print(accuracy)
# 0.625

# -----------------------------------
# logloss

from sklearn.metrics import log_loss

# 0, 1로 표현되는 2진분류의 실제값과 예측확률
y_true = [1, 0, 1, 1, 0, 1]
y_prob = [0.1, 0.2, 0.8, 0.8, 0.1, 0.3]

logloss = log_loss(y_true, y_prob)
print(logloss)
# 0.7136

# -----------------------------------
# 멀티클래스 분류
# -----------------------------------
# multi-class logloss

from sklearn.metrics import log_loss

# 3클래스 분류의 실제값과 예측값
y_true = np.array([0, 2, 1, 2, 2])
y_pred = np.array([[0.68, 0.32, 0.00],
                   [0.00, 0.00, 1.00],
                   [0.60, 0.40, 0.00],
                   [0.00, 0.00, 1.00],
                   [0.28, 0.12, 0.60]])
logloss = log_loss(y_true, y_pred)
print(logloss)
# 0.3626

# -----------------------------------
# 멀티라벨 분류
# -----------------------------------
# mean_f1, macro_f1, micro_f1

from sklearn.metrics import f1_score

# 멀티 라벨 분류의 실제값/예측값은 평가지표의 계산상은 레코드×클래스의 2값의 행렬로 하는 편이 다루기 쉽다
# 실제값 - [[1,2], [1], [1,2,3], [2,3], [3]]
y_true = np.array([[1, 1, 0],
                   [1, 0, 0],
                   [1, 1, 1],
                   [0, 1, 1],
                   [0, 0, 1]])

# 예측값 - [[1,3], [2], [1,3], [3], [3]]
y_pred = np.array([[1, 0, 1],
                   [0, 1, 0],
                   [1, 0, 1],
                   [0, 0, 1],
                   [0, 0, 1]])

# mean-f1에서는 레코드마다 F1-score을 계산하고 평균을 구한다
mean_f1 = np.mean([f1_score(y_true[i, :], y_pred[i, :]) for i in range(len(y_true))])

# macro-f1에서는 클래스마다 F1-score를 계산하고 평균을 구한다
n_class = 3
macro_f1 = np.mean([f1_score(y_true[:, c], y_pred[:, c]) for c in range(n_class)])

# micro-f1에서는 레코드×클래스의 페어마다 TP/TN/FP/FN을 계산하고 F1-score를 구한다
micro_f1 = f1_score(y_true.reshape(-1), y_pred.reshape(-1))

print(mean_f1, macro_f1, micro_f1)
# 0.5933, 0.5524, 0.6250

# scikit-learn의 메소드를 사용해서도 계산할 수 있다.
mean_f1 = f1_score(y_true, y_pred, average='samples')
macro_f1 = f1_score(y_true, y_pred, average='macro')
micro_f1 = f1_score(y_true, y_pred, average='micro')

# -----------------------------------
# 클래스간에 순서관계가 있는 멀티 클래스 분류
# -----------------------------------
# quadratic weighted kappa

from sklearn.metrics import confusion_matrix, cohen_kappa_score


# quadratic weighted kappa를 계산하는 함수
def quadratic_weighted_kappa(c_matrix):
    numer = 0.0
    denom = 0.0

    for i in range(c_matrix.shape[0]):
        for j in range(c_matrix.shape[1]):
            n = c_matrix.shape[0]
            wij = ((i - j) ** 2.0)
            oij = c_matrix[i, j]
            eij = c_matrix[i, :].sum() * c_matrix[:, j].sum() / c_matrix.sum()
            numer += wij * oij
            denom += wij * eij

    return 1.0 - numer / denom


# y_true는 실제값의 클래스 리스트, y_pred는 예측값의 클래스 리스트
y_true = [1, 2, 3, 4, 3]
y_pred = [2, 2, 4, 4, 5]

# 오차행렬을 계산한다
c_matrix = confusion_matrix(y_true, y_pred, labels=[1, 2, 3, 4, 5])

# quadratic weighted kappa를 계산한다
kappa = quadratic_weighted_kappa(c_matrix)
print(kappa)
# 0.6153

# scikit-learn의 메소드를 사용해서도 계산할 수 있다
kappa = cohen_kappa_score(y_true, y_pred, weights='quadratic')

# -----------------------------------
# 추천
# -----------------------------------
# MAP@K

# K=3, 레코드수는 5개, 클래스는 4종류로 한다.
K = 3

# 각 레코드의 실제값
y_true = [[1, 2], [1, 2], [4], [1, 2, 3, 4], [3, 4]]

# 각 레코드에 대한 예측값 - K=3이므로 일반적으로 각 레코드에 각각 3개까지 순위를 매겨 예측한다
y_pred = [[1, 2, 4], [4, 1, 2], [1, 4, 3], [1, 2, 3], [1, 2, 4]]


# 각 레코드마다 average precision을 계산하는 함수
def apk(y_i_true, y_i_pred):
    # y_pred가 K이하의 길이이고 요소가 전부 다른 경우가 필요
    assert (len(y_i_pred) <= K)
    assert (len(np.unique(y_i_pred)) == len(y_i_pred))

    sum_precision = 0.0
    num_hits = 0.0

    for i, p in enumerate(y_i_pred):
        if p in y_i_true:
            num_hits += 1
            precision = num_hits / (i + 1)
            sum_precision += precision

    return sum_precision / min(len(y_i_true), K)


# MAP@K 을 계산하는 함수
def mapk(y_true, y_pred):
    return np.mean([apk(y_i_true, y_i_pred) for y_i_true, y_i_pred in zip(y_true, y_pred)])


# MAP@K을 구하기
print(mapk(y_true, y_pred))
# 0.65

# 정답의 수가 같더라고 순서가 다르면 스코어도 다르다
print(apk(y_true[0], y_pred[0]))
print(apk(y_true[1], y_pred[1]))
# 1.0, 0.5833
