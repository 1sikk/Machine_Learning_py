import numpy as np
import sklearn
from sklearn.base import BaseEstimator
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import load_digits
from sklearn.metrics import confusion_matrix
from sklearn.metrics import  accuracy_score,precision_score,recall_score
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

class MyDummyClassfier(BaseEstimator):
    def fit(self,x,y=None):
        pass
    # fit 메서드에서는 아무것도 실행하지 않음
    #predict에서는 단순히 Sex가 1이면 0 아니면 1로 예측함
    def predict(self,X):
        predict = np.zeros((X.shape[0], 1))
        for i in range(X.shape[0]) :
            if X['Sex'].iloc[i] == 1 :
                predict[i] = 0
            else :
                predict[i] = 1

        return predict

### 내가만든 클래서파이를 활용하기위해 데이터 불러오기
def fillna(df):
    df['Age'].fillna(df['Age'].mean(), inplace=True) # Age컬럼의 결측값을 평균치로 대체
    df['Cabin'].fillna('N', inplace=True)
    df['Embarked'].fillna('N', inplace=True)
    df['Fare'].fillna(0, inplace=True)
    return df

# 사용되지않은는 컬럼 삭제
def drop_features(df):
    df.drop(['PassengerId','Name','Ticket'], axis=1, inplace=True)
    return df

def format_features(df):
    df['Cabin']= df['Cabin'].str[:1] #앞의 1음절만 slicing
    features = ['Cabin','Sex','Embarked']
    for feature in features:
        le = LabelEncoder()
        le = le.fit(df[feature])
        df[feature] = le.transform(df[feature])
    return df

def transform_featurs(df):
    df = fillna(df)
    df = drop_features(df)
    df = format_features(df)
    return df

titanic_df = pd.read_csv("/Users/sik/Desktop/CSV/titanic_train.csv")
y_titanic_df = titanic_df["Survived"]
X_titanic_df = titanic_df.drop('Survived', axis=1)
X_titanic_df = transform_featurs(X_titanic_df)
X_train, X_test, y_train, y_test = train_test_split(X_titanic_df, y_titanic_df, test_size=0.2, random_state=1) # 테스트 트레이닝셋 분리

dummy_clf = MyDummyClassfier()
dummy_clf.fit(X_train, y_train)

myprediction = dummy_clf.predict(X_test)
print('내 쓰레기 분류함수의 정확도는 : {0:.4f}'.format(accuracy_score(y_test,myprediction)))
#  이분적인 분류로 분류했을때도 77퍼센트라는 높은 정확도를 보인다.
# 정확도로는 분류하는데 문제가 있어보인다.

# MNIS데이터셋을 이진분류로 정확도 측정하기
class Mydummyclassifier2(BaseEstimator) :
    def fit(self,X,y):
        pass
    def predict(self, X):
        return np.zeros((len(X),1),dtype=bool)

digits = load_digits() #사이킷런의 내장데이터셋 안의 Mnist 로드
y = (digits.target == 7).astype(int)
X_train, X_test, y_train, y_test = train_test_split(digits.data, y, random_state=1)

print('레이블 테스트의 크기 :', y_test.shape)
print('테스트 세트 레이블 0과 1의 분포도 :')
print(pd.Series(y_test).value_counts()) # 0 402개, 1 48개

dummy_clf2 = Mydummyclassifier2()
dummy_clf2.fit(X_train, y_train)
dummy2_pred = dummy_clf2.predict(X_test)
print('모든 예측을 0으로 하였을때 정확도 : {0:.4f} :'.format(accuracy_score(y_test, dummy2_pred)))
# 정확도 90% 유수의 알고리즘의 성능을 내는 것을 볼수있는데 이 결과는 말도 안되는 결과이다.
# 정학도의 한계를 명확히 보여주는 사례이다.

# Confusion Matrix

confusion_matrix(y_test, dummy2_pred)
# confusion matrx , reall accuracy, precision을 한번에 나타내는 사용자 정의 함수 만들기

def get_clf_eval(y_test, pred):
    confusion = confusion_matrix(y_test, pred)
    accuracy= accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred)
    recall = recall_score(y_test,pred)
    print('오차 행렬 값')
    print(confusion)
    print('정확도 : {0:.4f}' '정밀도 : {1:.4f}' '재현율 : {2:.4f}'.format(accuracy,precision,recall))

# 오차행렬을 확인하기위한 데이터 로딩
titanic_df = pd.read_csv("/Users/sik/Desktop/CSV/titanic_train.csv")
y_titanic_df = titanic_df["Survived"]
X_titanic_df = titanic_df.drop('Survived', axis=1)
X_titanic_df = transform_featurs(X_titanic_df)
X_train, X_test, y_train, y_test = train_test_split(X_titanic_df, y_titanic_df, test_size=0.2, random_state=11) # 테스트 트레이닝셋 분리

#로지스틱 회귀분석
log_clf = LogisticRegression()

log_clf.fit(X_train, y_train)
predict = log_clf.predict(X_test)
get_clf_eval(y_test, predict)

#pred_proba를 잉요하여 임계값 조절하기

pred_proba = log_clf.predict_proba(X_test)
predict = log_clf.predict(X_test)
print('pred_proba의 결과 Shape : {0}'.format(pred_proba.shape)) # 179행 2열로 되어있음.
print('pred_proba array에서 앞 3개만 샘플로 추출 : \n', pred_proba[:3]) # 앞 3개만 호출

# 예측확률 array와 결과값의 array를 병합하여 예측결과와 결과값을 한번에 비교하기

pred_proba_result = np.concatenate([pred_proba, predict.reshape(-1,1)], axis=1)
print('두 개의 class 중에서 더큰 확률을 클래스 값으로 예측 \n',pred_proba_result[:3])

# binarze는 1과 같거나 작으면 0을 반환 크면 1을 반환한다.
# binarze를 이용하여 임계치를 부여하여 proba를 구할수 있음. 디폴트 임계치 0.5
from sklearn.preprocessing import Binarizer

x = [[1,-1,2],[2,0,0],[0.1,1.1,1.4]]

binarizer = Binarizer(threshold = 1.1)
print(binarizer.fit_transform(x))

# 설정 임계값
custom_threshold = 0.5
# predict_proba() 변환값의 두번재 칼럼, positive 클래스의 칼럼함 추출해 biniarizer 이용
pred_proba_1 = pred_proba[:,1].reshape(-1,1)

binarizer = Binarizer(threshold=custom_threshold).fit(pred_proba_1)
custom_predict = binarizer.transform(pred_proba_1)
get_clf_eval(y_test, custom_predict)

# 임계값을 0.4로 조정
custom_threshold = 0.4
# predict_proba() 변환값의 두번재 칼럼, positive 클래스의 칼럼함 추출해 biniarizer 이용
pred_proba_1 = pred_proba[:,1].reshape(-1,1)

binarizer = Binarizer(threshold=custom_threshold).fit(pred_proba_1)
custom_predict = binarizer.transform(pred_proba_1)
get_clf_eval(y_test, custom_predict)

# 여러 임계값을 정의하는 사용자 정의함수 만들기
# 임계값 0.6만 출력  ? Error ?  확인필요.
thresholds = [0.4,0.45,0.50,0.55,0.60]

def get_eval_by_threshold(y_test, pred_proba_c1, thresholds) :
    for custom_threshold in thresholds :
    binarizer = Binarizer(threshold=custom_threshold).fit(pred_proba_c1)
    custom_predict = binarizer.transform(pred_proba_c1)
    print('임계값:',custom_threshold)
    get_clf_eval(y_test, custom_predict)

get_eval_by_threshold(y_test, pred_proba[:,1].reshape(-1,1),thresholds)

# 사이킷런에는 위의 사용자 정의함수와 유사한 API를 지원한다.

# 레이블 값이 1일 때의 예측 확률을 추출
pred_proba_class1 = log_clf.predict_proba(X_test)[:,1]
# 실제값 데이터 세트와 레이블 값이 1일 때의 예측확률을 precision_recall_curve의 인자로 입력하기
precisoions, recalls, thresholds = precision_recall_curve(y_test, pred_proba_class1)
print('반환된 분류 결정 임곗값 배열의 shape : ', thresholds.shape) # precision_recall_curve는 0.11~ 0.95사이의 데이터값을 담고있는 ndarray로 구성됨
# 반환된 임계값이 143개나 되기때문에 10개정도만 비교하기로하되 15개의 step을 둠
thr_index = np.arange(0,thresholds.shape[0],15)
print('샘플 추출을 위한 임계값 배열의 index 10개 : ', thr_index # 15개의 스탭으로 분류된 인덱스를 볼수 있음.
print('샘플용 10개의 임계값 : ',np.round(thresholds[thr_index],2)) # 분류된 임계값의 소숫점 2자리수 까지 확인
# 15 스탭으로 추출된 임계값의 정밀도와 재현율 값
print('임계값 별 정밀도 : ', np.round(precisoions[thr_index],3))
print('임계값 별 재현율 : ', np.round(recalls[thr_index],3))

# 임계값 별 변화를 시각적으로 볼수 있는 그래프 만들기
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def precision_recall_curve_plot(y_test,pred_proba_c1) :
    # threshold ndarray와 threshold에 따른 정밀도, 재현율 ndarray 추출
    precisions, recalls, thresholds = precision_recall_curve(y_test, pred_proba_c1)
    # x 축은 임계값으로 , y축은 정밀도, 재현율 값으로 각각 plot를 수행 정밀도는 점선으로 표시
    plt.figure(figsize=(8,6))
    threshole_boundary = thresholds.shape[0]
    plt.plot(thresholds, precisions[0:threshole_boundary], linestyle='--', label='precision')
    plt.plot(thresholds, recalls[0:threshole_boundary],label='recall')
    # threshold 값 X축의 크기을 0.1단위로 변경
    start, end = plt.xlim()
    plt.xticks(np.round(np.round(start,end,0.1),2))
    # x,y축의 이름과 , 격자무늬 생성하기
    plt.xlabel('Threshold value'); plt.ylabel('Precision and Recall value') # x축과 y축 이름 생성
    plt.legend(); plt.gird()
    plt.show()

precision_recall_curve_plot(y_test, log_clf.predict_proba(X_test)[:,1])

## F1 스코어
# F1 스코어는 정밀도와 재현율을 결합한 지표이다. 정밀도와 재현율이 어느쪽에 치우치지않을때 높은 값을 갖는다.

from sklearn.metrics import f1_score
f1 = f1_score(y_test, predict)
print('F1스코어:{0:.4f}'.format(f1))

# 임계치를 조정하면서 F1 지표도 같이보기
def get_clf_eval_threshold(y_test, pred):
    confusion = confusion_matrix(y_test, pred)
    accuracy= accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred)
    recall = recall_score(y_test,pred)
    # f1 스코어도 추가
    f1 =  f1_score(y_test, pred)
    print('오차 행렬 값')
    print(confusion)
    print('정확도 : {0:.4f}' '정밀도 : {1:.4f}' '재현율 : {2:.4f}' 'F1 score : {3:.4f}'.format(accuracy,precision,recall,f1))

thresholds = [0.4, 0.45, 0.5, 0.55, 0.60]
pred_proba = log_clf.predict_proba(X_test)
get_eval_by_threshold(y_test, pred_proba[:,1].reshape(-1,1), thresholds)
