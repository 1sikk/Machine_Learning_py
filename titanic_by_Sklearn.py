# 사이킷런으로 타이타닉호 생존자 예측하기 (kaggle)
import matplotlib.pyplot as plt  # 시각화를 위해 필요한 패키지
import numpy as np  # 넘파이
import pandas as pd  # 판다스
import seaborn as sns  # 시각화 패키지
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score # 정확도 측정

# 타이타닉 데이터에 대한 주석
# Passengerid : 탑승자 데이터 일련번호
# survived : 생존여부 0 = 사망, 1 = 생존
# pclass = 티켓 선실 등급 1 = 일등석 2 = 이등석 3 = 삼등석
# sex : 성별
# name : 이름
# Age :  탑승자 나이
# sibsp : 같이 탑승한 형제자매 또는 배우자 인원수
# parch : 같이 탑승한 부모님 또는 어린이 인원수
# ticket : 티켓번호
# fare : 요금
# cabin : 선실 번호
# embarked : 중간 정착 항구 C = 셰르부르 , Q = 퀸즈타운, S= 사우스햄튼

# 타이타닉 데이터셋 로드
titanic_df = pd.read_csv('/Users/sik/Desktop/CSV/titanic_train.csv')
titanic_df.head(3)

# 데이터 컬럼 타입 확인
print('\n ### 학습 데이터 정보 ### \n')
print(titanic_df.info)

# 1. 사이킷런 머신러닝에서는 결측값이 있으면 안되므로 처리해준다.
titanic_df.isna().sum()  # Age에 177개  Cabin 687개 Embarked에 2개의 결측값을 가지고 있는것을 볼 수 있다.
titanic_df['Age'].fillna(titanic_df['Age'].mean(), inplace=True)  # 나이의 평균값으로 결측값 채워 넣기
titanic_df['Cabin'].fillna('N', inplace=True)  # 'N'을 결측값으로채워 넣기
titanic_df['Embarked'].fillna('N', inplace=True)  # 'N'을 결측값으로 채워 넣기

# 2. 값의 분류값들을 확인 해주기
print('Sex 값의 분포 : \n', titanic_df['Sex'].value_counts())
print('\n cabin 값 분포 : \n', titanic_df['Cabin'].value_counts())
print('\nEmbarked 값 분포 : \n', titanic_df['Embarked'].value_counts())
# Sex와 Cabin의 데이터 분포는 정리가 잘되어있지만
# Embarked의 데이터분포는 다소 난해하게 정리가 되어 있는 것을 볼 수 있다.
# 이시절에는 빈부 격차에 따라 사람은 차별하는것이 더 심했을것으로 생각이 된다 따라서 객실의 등급은 중요한 정보이다. (드랍해서는 안됨)
titanic_df['Cabin'] = titanic_df['Cabin'].str[:1]
print(titanic_df['Cabin'].head(3))
# 성별을 기준으로 생존률에 대한 카운트 이다.
# 사고가 발생했을때 여자와 어린아이를 먼저 대피시키기때문에 이와 밀접한 관계가 있을 것으로 생각이된다.
titanic_df.groupby(['Sex', 'Survived'])['Survived'].count()
# 데이터를 시각화하여 조금더 편하게 확인하기
sns.barplot(x='Sex', y='Survived', data=titanic_df)
# 여자는 약 70퍼센트정도 생존했지만 남자의경우 20에 못미치는것을 알수 있다.
# 부유한사람과 가난한사람의 생존률 비교 (hue라는 파라미터로 편하게 성별을 구분할수 있다)
sns.barplot(x='Pclass', y='Survived', hue='Sex', data=titanic_df)


# 1등급과 2등급의 여성의 생존률 차이는 적지만 3등급으로 떨어질경우 생존률이 급격하게 차이나는 것을 볼수 있다 .
# 나이에 따라 카테고리를 나눠주는 사용자 정의 함수 만들어주기
def get_category(age):
    cat = ''
    if age <= -1:
        cat = 'Strange'
    elif age <= 5:
        cat = 'Baby'
    elif age <= 13:
        cat = 'Child'
    elif age <= 19:
        cat = 'Teenager'
    elif age <= 25:
        cat = 'Blue young Adult'
    elif age <= 35:
        cat = 'Young Adult'
    elif age <= 60:
        cat = 'Adult'
    else:
        cat = 'Elderly'

    return cat

plt.Figure(figsize=(10, 6))

# X축의 값을 순서대로 표시하기위한 설정
group_names = ['Strange', 'Baby', 'Child', 'Teenager', ' Blue young Adult', 'Young Adult', 'Adult', 'Elderly']
titanic_df['Age_category'] = titanic_df['Age'].apply(lambda x: get_category(x))
sns.barplot(x='Age_category', y='Survived', hue='Sex', data=titanic_df, order=group_names)
titanic_df.drop('Age_category', axis=1, inplace=True)

#3. 문자열 카테고리 피쳐를 숫자형으로 변환해주기

def encode_feature(x):
    features = ['Cabin','Sex','Embarked']
    for feature in features:
        le = preprocessing.LabelEncoder()
        le = le.fit(x[feature])
        x[feature] = le.transform(x[feature])

    return x

titanic_df = encode_feature(titanic_df)
titanic_df.head() # 숫자형으로 레이블 인코딩 된것을 볼수 있다.

#위에서 사용된 모든 전처리과정을 한번에 처리할수 있는 사용자 함수를 만들어서 사용하기

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

# 생존률을 피쳐 데이터 세트로 가공하기
titanic_df = pd.read_csv('/Users/sik/Desktop/CSV/titanic_train.csv')
y_titan_df = titanic_df['Survived']
X_titan_df = titanic_df.drop('Survived', axis=1)
X_titan_df = transform_featurs(X_titan_df)
X_titan_df
y_titan_df
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_titan_df, y_titan_df, test_size=0.2, random_state=11)

# 사용할 알고리즘 임포트
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score # 정확도 측정

dt_titanic = DecisionTreeClassifier(random_state=11)
rf_titanic = RandomForestClassifier(random_state=1)
lr_titanic = LogisticRegression()

# 의사결정나무 학습하고 예측 평가하기
dt_titanic.fit(X_train, y_train)
dt_pred = dt_titanic.predict(X_test)
print('의사 결정 나무의 예측 정확도 : {0:0.4f}'.format(accuracy_score(y_test, dt_pred))) #예측정확도 0.7374

# 랜덤포레스트 학습하고 예측 평가하기
rf_titanic.fit(X_train,y_train)
rf_pred = rf_titanic.predict(X_test)
print('랜덤 포레스트의 예측 정확도 :{0:0.4f}'.format(accuracy_score(y_test,rf_pred))) # 예측정확도 0.7654

# 로지스틱회귀분석 학습하고 예측 평가하기
lr_titanic.fit(X_train, y_train)
lr_pred = lr_titanic.predict(X_test)
print('로지스틱 회귀분석의 정확도 : {0:0.4f}'.format(accuracy_score(y_test,lr_pred))) # 예측정확도 0.7988

#로지스틱 회귀분석이 다른 데이터에비해서 높은 정확도를 보여주고 있지만 데이터의 크기가 작아 확실히 좋은 방법이라고는 할 수 없다.

# KFold를 이용하여 교차검증 하기
from sklearn.model_selection import KFold

def exec_kfold(clf, folds=5) :
    kfold = KFold(n_splits=folds)
    scores = []

    for iter_count, (train_index, test_index) in enumerate(kfold.split(X_titan_df)):
        X_train,X_test = X_titan_df.values[train_index], X_titan_df.values[test_index]
        y_train, y_test = y_titan_df.values[train_index], y_titan_df.values[test_index]

        clf.fit(X_train,y_train)
        predict = clf.predict(X_test)
        accuracy = accuracy_score(y_test, predict)
        scores.append(accuracy)
        print('교차 검증{0}번 정확도 : {1:.4f}'.format(iter_count,accuracy))

    mean_score = np.mean(scores)
    print('평균 정확도 : {0:.4f}'.format(mean_score))

exec_kfold(dt_titanic, folds=5)

# cross_val_score() API를 이용하여 수행하기
from sklearn.model_selection import cross_val_score
scores = cross_val_score(dt_titanic, X_titan_df, y_titan_df, cv = 5) # 5번수행
for iter_count, accuracy in enumerate(scores):
    print('교차 검증 {0}번, 정확도 : {1:.4f}'.format(iter_count, accuracy))

print("평균 정확도 : {0:.4f}".format(np.mean(scores)))
# 정확도에 차이가나는 이유는 cross_val_score가 startifiedKFold를 이용하여 세트를 분할하기 때문이다.

# GridSearchCV를 이용해 최적의 하이퍼 파라미터 찾기
from sklearn.model_selection import GridSearchCV


dt_titanic = DecisionTreeClassifier(random_state=1)
parameters = {'max_depth':[2,3,5,10], 'min_samples_split':[2,3,5], 'min_samples_leaf':[1,5,8]}
grid_dtlf= GridSearchCV(dt_titanic, param_grid=parameters, scoring='accuracy', cv=5)
grid_dtlf.fit(X_train, y_train)

print('최적의 하이퍼 파라미터 : ', grid_dtlf.best_params_)
print('최고 정확도 : {0:.4f}'.format(grid_dtlf.best_score_))
best_dtlf = grid_dtlf.best_estimator_

#최적의 파라미터로 학습하고 예측하기
best_pred = best_dtlf.predict(X_test)
accuracy = accuracy_score(y_test, best_pred)
print('최적의 파라미터로 튜닝후 예측 정확도 : {0:.4f}'.format(accuracy))
# 8퍼센트 상향된 결과를 보이지만 데이터가 많은 경우에는 무의미한 수치가 오른다고 함.
