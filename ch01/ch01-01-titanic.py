import numpy as np
import pandas as pd

# -----------------------------------
# 학습데이터, 테스트데이터 읽어들이기
# -----------------------------------
# 학습데이터, 테스트데이터 읽어들이기
train = pd.read_csv('../input/ch01-titanic/train.csv')
test = pd.read_csv('../input/ch01-titanic/test.csv')

# 학습데이터를 특징량과 목적변수로 나누기
train_x = train.drop(['Survived'], axis=1)
train_y = train['Survived']

# 테스트데이터는 특징량만 있기에 이대로 둠
test_x = test.copy()

# -----------------------------------
# 특징량 작성
# -----------------------------------
from sklearn.preprocessing import LabelEncoder

# 변수PassengerId를 제외한다.
train_x = train_x.drop(['PassengerId'], axis=1)
test_x = test_x.drop(['PassengerId'], axis=1)

# 변수Name, Ticket, Cabin을 제외한다.
train_x = train_x.drop(['Name', 'Ticket', 'Cabin'], axis=1)
test_x = test_x.drop(['Name', 'Ticket', 'Cabin'], axis=1)

# 각각의 범주변수에 label encoding을 적용한다.
for c in ['Sex', 'Embarked']:
    # 학습 데이터에 기반하여 어떻게 변환할 지를 정한다.
    le = LabelEncoder()
    le.fit(train_x[c].fillna('NA'))

    # 학습데이터, 테스트데이터를 변환한다.
    train_x[c] = le.transform(train_x[c].fillna('NA'))
    test_x[c] = le.transform(test_x[c].fillna('NA'))

# -----------------------------------
# 모델작성
# -----------------------------------
from xgboost import XGBClassifier

# 모델 작성 및 학습데이터를 학습
model = XGBClassifier(n_estimators=20, random_state=71)
model.fit(train_x, train_y)

# 테스트데이터의 예측값을 확률로 출력한다.
pred = model.predict_proba(test_x)[:, 1]

# 테스트데이터의 예측값을 2진값으로 변환한다.
pred_label = np.where(pred > 0.5, 1, 0)

# 제출용 파일 작성
submission = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': pred_label})
submission.to_csv('submission_first.csv', index=False)
# 스코어：0.7799（책과 수치가 다를 가능성이 있습니다）

# -----------------------------------
# 밸리데이션(검증)
# -----------------------------------
from sklearn.metrics import log_loss, accuracy_score
from sklearn.model_selection import KFold

# 각fold의 스코어를 저장하는 리스트
scores_accuracy = []
scores_logloss = []

# 크로스밸리데이션을 실행한다.
# 학습데이터를 4개로 분할하고 그 중 하나를 밸리데이션데이터로 하고 밸리데이션 데이터를 바꿔가며 반복한다.
kf = KFold(n_splits=4, shuffle=True, random_state=71)
for tr_idx, va_idx in kf.split(train_x):
    # 학습데이터를 학습데이터와 밸리데이션데이터로 나눈다.
    tr_x, va_x = train_x.iloc[tr_idx], train_x.iloc[va_idx]
    tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]

    # 모델의 학습을 진행한다.
    model = XGBClassifier(n_estimators=20, random_state=71)
    model.fit(tr_x, tr_y)

    # 밸리데이션데이터의 예측값을 활률로 출력한다.
    va_pred = model.predict_proba(va_x)[:, 1]

    # 밸리데이션데이터로 스코어를 계산한다.
    logloss = log_loss(va_y, va_pred)
    accuracy = accuracy_score(va_y, va_pred > 0.5)

    # 그 fold의 스코어를 저장한다.
    scores_logloss.append(logloss)
    scores_accuracy.append(accuracy)

# 각 fold의 스코어의 평균을 출력한다.
logloss = np.mean(scores_logloss)
accuracy = np.mean(scores_accuracy)
print(f'logloss: {logloss:.4f}, accuracy: {accuracy:.4f}')
# logloss: 0.4270, accuracy: 0.8148（책과 수치가 다를 가능성이 있습니다）

# -----------------------------------
# 모델 튜닝
# -----------------------------------
import itertools

# 튜닝 후보로 할 파라메터를 준비한다.
param_space = {
    'max_depth': [3, 5, 7],
    'min_child_weight': [1.0, 2.0, 4.0]
}

# 탐색할 하이퍼파라메터의 조합
param_combinations = itertools.product(param_space['max_depth'], param_space['min_child_weight'])

# 각 파라메터의 조합에 대한 스코어를 저장하는 리스트
params = []
scores = []

# 각 파라메서 조합마다 크로스밸리데이션으로 평가를 진행한다.
for max_depth, min_child_weight in param_combinations:

    score_folds = []
    # 크로스밸리데이션을 진행한다.
    # 학습데이터를 4개로 분할하고 그 중 하나를 밸리데이션데이터로 하고 밸리데이션 데이터를 바꿔가며 반복한다.
    kf = KFold(n_splits=4, shuffle=True, random_state=123456)
    for tr_idx, va_idx in kf.split(train_x):
        # 학습데이터를 학습데이터와 밸리데이션데이터로 나눈다.
        tr_x, va_x = train_x.iloc[tr_idx], train_x.iloc[va_idx]
        tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]

        # 모델의 학습을 진행한다.
        model = XGBClassifier(n_estimators=20, random_state=71,
                              max_depth=max_depth, min_child_weight=min_child_weight)
        model.fit(tr_x, tr_y)

        # 밸리데이션데이터로 스코어를 계산하고 저장한다.
        va_pred = model.predict_proba(va_x)[:, 1]
        logloss = log_loss(va_y, va_pred)
        score_folds.append(logloss)

    # 각 fold의 스코어의 평균을 계산한다.
    score_mean = np.mean(score_folds)

    # 파라메터의 조합에 대한 스코어를 저장한다.
    params.append((max_depth, min_child_weight))
    scores.append(score_mean)

# 가장 스코어가 좋은 것을 베스트 파라메터로 한다.
best_idx = np.argsort(scores)[0]
best_param = params[best_idx]
print(f'max_depth: {best_param[0]}, min_child_weight: {best_param[1]}')
# max_depth=7, min_child_weight=2.0 의 스코어가 가장 좋았다.


# -----------------------------------
# 로지스틱회귀용 특징량 작성
# -----------------------------------
from sklearn.preprocessing import OneHotEncoder

# 원데이터를 복사한다.
train_x2 = train.drop(['Survived'], axis=1)
test_x2 = test.copy()

# 변수PassengerId를 제외한다.
train_x2 = train_x2.drop(['PassengerId'], axis=1)
test_x2 = test_x2.drop(['PassengerId'], axis=1)

# 변수Name, Ticket, Cabin을 제외한다.
train_x2 = train_x2.drop(['Name', 'Ticket', 'Cabin'], axis=1)
test_x2 = test_x2.drop(['Name', 'Ticket', 'Cabin'], axis=1)

# one-hot encoding을 실행한다.
cat_cols = ['Sex', 'Embarked', 'Pclass']
ohe = OneHotEncoder(categories='auto', sparse=False)
ohe.fit(train_x2[cat_cols].fillna('NA'))

# one-hot encoding의 더미변수의 열명을 작성한다.
ohe_columns = []
for i, c in enumerate(cat_cols):
    ohe_columns += [f'{c}_{v}' for v in ohe.categories_[i]]

# one-hot encoding에 의한 변환을 실행한다.
ohe_train_x2 = pd.DataFrame(ohe.transform(train_x2[cat_cols].fillna('NA')), columns=ohe_columns)
ohe_test_x2 = pd.DataFrame(ohe.transform(test_x2[cat_cols].fillna('NA')), columns=ohe_columns)

# one-hot encoding이 완료된 변수를 제외한다.
train_x2 = train_x2.drop(cat_cols, axis=1)
test_x2 = test_x2.drop(cat_cols, axis=1)

# one-hot encoding으로 변환된 변수를 결합한다.
train_x2 = pd.concat([train_x2, ohe_train_x2], axis=1)
test_x2 = pd.concat([test_x2, ohe_test_x2], axis=1)

# 수치변수의 결측값을 학습데이터의 평균으로 채운다.
num_cols = ['Age', 'SibSp', 'Parch', 'Fare']
for col in num_cols:
    train_x2[col].fillna(train_x2[col].mean(), inplace=True)
    test_x2[col].fillna(train_x2[col].mean(), inplace=True)

# 변수Fare를 log로 변환한다.
train_x2['Fare'] = np.log1p(train_x2['Fare'])
test_x2['Fare'] = np.log1p(test_x2['Fare'])

# -----------------------------------
# 앙상블
# -----------------------------------
from sklearn.linear_model import LogisticRegression

# xgboost모델
model_xgb = XGBClassifier(n_estimators=20, random_state=71)
model_xgb.fit(train_x, train_y)
pred_xgb = model_xgb.predict_proba(test_x)[:, 1]

# 로지스틱회귀모델
# xgboost모델과는 다른 특징량을 넣을 필요가 있기 때문에 별도의 train_x2, test_x2를 작성했다.
model_lr = LogisticRegression(solver='lbfgs', max_iter=300)
model_lr.fit(train_x2, train_y)
pred_lr = model_lr.predict_proba(test_x2)[:, 1]

# 예측값의 가중평균을 구한다.
pred = pred_xgb * 0.8 + pred_lr * 0.2
pred_label = np.where(pred > 0.5, 1, 0)
