# 사이킷런으로 타이타닉호 생존자 예측하기 (kaggle)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

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
titanic_df.isna().sum() # Age에 177개  Cabin 687개 Embarked에 2개의 결측값을 가지고 있는것을 볼 수 있다.
titanic_df['Age'].fillna(titanic_df['Age'].mean(), inplace=True) # 나이의 평균값으로 결측값 채워 넣기
titanic_df['Cabin'].fillna('N',inplace=True) # 'N'을 결측값으로채워 넣기
titanic_df['Embarked'].fillna('N', inplace=True) #'N'을 결측값으로 채워 넣기

# 2. 값의 분류값들을 확인 해주기
print('Sex 값의 분포 : \n', titanic_df['Sex'].value_counts())
print('\n cabin 값 분포 : \n',titanic_df['Cabin'].value_counts())
print('\nEmbarked 값 분포 : \n',titanic_df['Embarked'].value_counts())
# Sex와 Cabin의 데이터 분포는 정리가 잘되어있지만
# Embarked의 데이터분포는 다소 난해하게 정리가 되어 있는 것을 볼 수 있다.
# 이시절에는 빈부 격차에 따라 사람은 차별하는것이 더 심했을것으로 생각이 된다 따라서 객실의 등급은 중요한 정보이다. (드랍해서는 안됨)
titanic_df['Cabin'] = titanic_df['Cabin'].str[:1]
print(titanic_df['Cabin'].head(3))
# 성별을 기준으로 생존률에 대한 카운트 이다.
# 사고가 발생했을때 여자와 어린아이를 먼저 대피시키기때문에 이와 밀접한 관계가 있을 것으로 생각이된다.
titanic_df.groupby(['Sex','Survived'])['Survived'].count()
# 데이터를 시각화하여 조금더 편하게 확인하기
sns.barplot(x='Sex',y='Survived', data = titanic_df)
# 여자는 약 70퍼센트정도 생존했지만 남자의경우 20에 못미치는것을 알수 있다.
# 부유한사람과 가난한사람의 생존률 비교 (hue라는 파라미터로 편하게 성별을 구분할수 있다)
sns.barplot(x='Pclass', y='Survived',hue='Sex',data=titanic_df)
# 1등급과 2등급의 여성의 생존률 차이는 적지만 3등급으로 떨어질경우 생존률이 급격하게 차이나는 것을 볼수 있다 .
# 나이에 따라 카테고리를 나눠주는 사용자 정의 함수 만들어주기
def get_category(age):
    cat = ''
    if age <= -1: cat = 'Strange'
    if age <= 5: cat = 'Baby'
    if age <= 13: cat = 'Child'
    if age <= 19: cat = 'Teenager'
    if age <= 25: cat = 'Blue young Adult'
    if age <= 35: cat = 'Young Adult'
    if age <= 60: cat = 'Adult'
    else: cat = 'Elderly'
    return cat

plt.Figure(figsize=(10,6))

#X축의 값을 순서대로 표시하기위한 설정
group_names = ['Unknown', 'Baby', 'Child', 'Teenager',' Blue young Adult', 'Young Adult', 'Adult', 'Elderly']

titanic_df['Age_category'] = titanic_df['Age'].apply(lambda x: get_category(x))
sns.barplot(x='Age_category', y='Survived', hue='Sex', data=titanic_df, order=group_names)
titanic_df.drop('Age_category', axis=1, inplace=True)