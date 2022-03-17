# 사이킷런에서 레이블 인코딩은 LableEncodr 클래스로 구현함.
import sklearn
from sklearn.preprocessing import LabelEncoder
import graphviz
cars = ['소나타','람보르기니','아반떼','테슬라','테슬라','그랜져','그랜져']

encoder = LabelEncoder()
encoder.fit(cars)
labels = encoder.transform(cars)
print('인코딩 변환값 : ', labels)  # 데이터가 작은경우 직관적으로 볼수있어서 편하지만 데이터가 많을 경우 혼돈 이있을수있다.
# 인코딩 한 결과 원본값 보기
print('디코딩 원본값 : ', encoder.inverse_transform([2,1,3,4,4,0,0]))
# 레이블 인코딩의 경우에는 회귀나 머신러닝 알고리즘 에서는 가중치 부여 되거나 더 중요하다고 인식 할 수도 있기 때문에 적용 하지 않아야 한다.
# 트리의 경우는 레이블 인코딩을 사용해도 무방하다.

# 원 핫 인코딩
# 원 핫 인코딩은 피쳐값의 유형에 따라 새로운 피처를 추가해서 고유값에 해당하는 칼럼에만 1을 표시하고 나머지 칼럼에는 0을 표기하는 방식을 말한다.

from sklearn.preprocessing import OneHotEncoder
import numpy as np

cars = ['소나타','람보르기니','아반떼','테슬라','테슬라','그랜져','그랜져']
# 원 핫인코딩을 하기위해서는 레이블 코드로 변경이 필요하다.
encoder = LabelEncoder()
encoder.fit(cars)
labels = encoder.transform(cars)
# 만들어진 라벨을 2차원 값으로 만듬
labels = labels.reshape(-1, 1)
# 원 핫 인코딩을 적용하기
one_encoder = OneHotEncoder()
one_encoder.fit(labels)
one_labels = one_encoder.transform(labels)
print('원 핫 인코딩 데이터 :')
print(one_labels.toarray())
print('원 핫 인코딩 데이터 차원 :') # 7행 5열로 이루어진것을 알수 있음.
print(one_labels.shape)

# sklean보다 편하게 원핫 인코딩하기 (두번의 절차를 걸쳐야함.)
# pandas의 get_dummy를 이용하면 좀더 편하게 데이터를 원 핫 인코딩이 가능하다
import pandas as pd
df= pd.DataFrame({'cars':['소나타','람보르기니','아반떼','테슬라','테슬라','그랜져','그랜져']})
pd.get_dummies(df)

# 피처 스케일링
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
import pandas as pd

iris = load_iris()
iris_data = iris.data
iris_df = pd.DataFrame(data=iris_data, columns=iris.feature_names)

print('feature 들의 평균 값')
print(iris_df.mean())
print('\nfeature 들의 분산 값 :')
print(iris_df.var())
# 평균과 분산값이 제각각 인것을 볼수 있다.
# StandardScale 표준화 가우시안 정규분포를 가질수있도록 데이터를 변환해줌
# StandardScale 객체를 생성하기
scale = StandardScaler()
# StandardScaler로 데이터 세트 변환
scale.fit(iris_df)
iris_scaled = scale.transform(iris_df)
# 트랜스폼을 하게되면 데이터의 형태가 ndarray로 변환이 되는데 이를 Dataframe 으로 변환을 해주어야함 .
iris_df_scaled = pd.DataFrame(data=iris_scaled, columns=iris.feature_names)
print('feature의 평균값 :')
print(iris_df_scaled.mean())
print('\nfeature들의 분산 값')
print(iris_df_scaled.var())
# 모든 칼럼값을 평균이 0 분산은 1에  아주 가까운 값으로 변환 되었 음을 알 수 있다.

# MinMaxScaler
from sklearn.preprocessing import MinMaxScaler

scale = MinMaxScaler()
scale.fit(iris_df)
iris_scaled = scale.transform(iris_df)

iris_df_scaled = pd.DataFrame(data = iris_scaled, columns=iris.feature_names)
print('Feature의 최솟값 :')
print(iris_df_scaled.min())
print('Feature의 최댓값 : ')
print(iris_df_scaled.max())


