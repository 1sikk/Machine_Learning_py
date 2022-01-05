import os
os.getcwd()
os.chdir('/Users/sik/Desktop/CSV')
import pandas as pd

# 판다스를 이용하여 데이터 프레임 생성 csv 말고 다른 파일도 로딩가능
# 판다스는 기본적으로 ndarray 의 값을 가지고 있음
titan_df = pd.read_csv("titanic_train.csv")
print(titan_df)

# 5번째 데이터 까지만 보기 (디폴트 값은 5)
titan_df.head(3)

# 타입확인
# 데이터 프레임으로 생성된것을 볼수있음.
print("titan_df 의 데이터 타입은 : ", type(titan_df))

# 행과 열의 크기를 가장 잘 확인 할수있는 방법은 .shape를 이용한다.
print(("titan_df 의 행과열을 쉽게 확인하기 : ", titan_df.shape)) # 891개의 행과 12개의 컬럼을 가진것을 확인 할수 있다.

# 데이터프레임의 상세 메타데이터 확인하기
print("titan_df의 메타 데이터 확인하기 : ",  titan_df.info()) # 컬럼명 , NULL의수, 데이터 타입을 확인 할수 있음

# 데이터값에 이상치가 많으면 예측성능이 저하됨. 간단히 .describe로 확인이 가능함
print("titan_df의 분포도 확인하기 : ", titan_df.describe()) # 카운트,평균,표준편차,최소값,4분위수,최대값을 확인할수 있음.

# 데이터 프레임에 []연산자 내부에 컬럼명을 입력하게되면 특정 칼럼 데이터 세트가 변환됨
# 이렇게 반환된 특정 컬럼 데이터 세트는 특별한 메서드를 사용이 가능하다.
value_counts = titan_df['Pclass'].value_counts() # 데이터 3을가진 행은 491행 , 데이터 1을 가진값은 216개 , 데이터 2를가진 행은 184개임을 알수 있음 데이터타입은 정수형
print(value_counts)
# 시리즈의 시리즈 형임을 볼수있음
titnic_pcalss = titan_df['Pclass']
print(type(titnic_pcalss))
titnic_pcalss.head()

# 1차원의 ndarray , 리스트, 딕셔너리를 dataframe으로 변환하기

import numpy as np

## 1차원의 데이터 데이터 프레임으로 변환하기
# 1차원의 경우는 데이터가하나면 데이터 프레임 변환이가능하다.
col_name1 = ['col1']
list1 = [1,2,3]
array1 = np.array(list1)
print('array1 의 타입은 : ', array1.shape) # 1차원의 ndarray가 생성됨을 볼수 있다.
# 위에 생성한 리스트로 데이터 프레임으로 변환하기
df_list = pd.DataFrame(list1, columns=col_name1)
print('df_list의 데이터 타입은 :', type(df_list))
print('1차원 리스트를 데이터 프레임으로 변환:\n', df_list)
# ndarray를 데이터 프레임으로 변환하기
df_array = pd.DataFrame(array1, columns=col_name1)
print(type(df_array))
print('ndarray을 데이터 프레임으로 변환하기 : \n', df_array)

## 2차원의 데이터 프레임으로 변환하기
#컬럼 3개 생성
col_name2 = ['col1','col2','col3']
list2 = [[1,2,3],[4,5,6]]
array2 = np.array(list2)
print(array2)
print('array2 의 형태 :', array2.shape)
df_list2 = pd.DataFrame(list2,columns=col_name2)
print('2차원리스트 -> 데이터 프레임 : \n', df_list2)
df_array2 = pd.DataFrame(array2, columns= col_name2)
print('2차원의 ndarray -> Data frame : \n', df_array2)

# dic형을 데이터 프레임으로 변경하기
dict = {'col1':[1,11],'col2':[2,22],'col3':[3,33]}
df_dict = pd.DataFrame(dict)
print('dic -> Dataframe : \n', df_dict)
# 딕트형을 데이터 프레임으로 변경할 경우에는 Key값은 컬럼명 Value는 칼럼의 데이터로 변경되는것을 볼수있음

# 데이터 프레임을 ndarray로 변경하기
array3 =  df_dict.values # 많이 사용되니 기억 잘해둘것
print('df_dict.values의 타입 : ', type(array3) ,'df_dict의 shape : ', array3.shape) # 데이터 프레임에서 ndarray로 변경된것을 볼수있음
# 데이터 프레임에서 리스트로 변경하기
list = array3.tolist()
print('DF -> ndarray -> list : \n', list) # DF형태에서 ndarray로 .values를 이용하여 변경한뒤 .tolist()를 이용하여 리스트화 할수있다.
# 데이터프레임에서 딕트형으로 변경하기
dict = df_dict.to_dict('list') # 리스트 형태의 딕트로 반환된 것을 볼수 있음
print('DF -> Dic형으로 변환 : \n', dict)

## 데이터 프레임 칼럼 데이터 세트 생성과 수정하기

## 생성
# zero_age 라는 컬럼을 생성하고 0을 데이터로 넣기
titan_df['zero_age']=0
titan_df.head()
# 컬럼의 데이터값을 이용하여 새로운 컬럼 만들기
titan_df['Age*10'] = titan_df['Age']*10 # Age*10이라는 컬럼을 생성하고 Age컬럼에 있는 데이터를 *10 해서 데이터를 추가
titan_df['Family_No'] = titan_df['SibSp'] + titan_df['Parch']+1 # Family_No라는 컬럼을 생성하고 SibSp 컬럼과 Parch라는 컬럼을 더하고 +1한 데이터를 추가
titan_df.head(3)

## 삭제
# drop을 이용하여 삭제가 가능하다 .
# DataFrame(labels=None, axis = 0(행) or 1(컬럼), index = None , columns=None, level=None,
#           inplace=False(원본데이터값 수정 x , True로 설정시 원본 데이터 값을 수정한다.), errors='raise)
# axis , inplace 메서드를 잘 확인하여야함.

# 이전에 생성된 zero_age 컬럼을 삭제하기
titan_df_drop = titan_df.drop('zero_age', axis=1)
titan_df_drop.head() # zero_age 컬럼이 삭제된것을 확인할수있음
titan_df.head()# inplace= False 즉 디폴트로 드롭이 되었기때문에 원본 데이터에서는 삭제 되지 않을 것을 알수 있다.
drop_titan = titan_df.drop(['zero_age','Age*10','Family_No'], axis=1, inplace=True) # 생성했던 컬럼들이 모두 삭제된 것을 확인 할 수 있다.
titan_df.head()

# 인덱스추출
# titan data reroding
index = titan_df.index
print(index)
# 실제 인댁스 객체값 array로 변환
print('index 객체 array값:\n', index.values)
# 인덱싱 객체는 1차원의 array 형태를 띄고 있음
print(type(index.values)) # ndarray형태임을 알 수 있음
print(index.values.shape) # 891개의 데이터를 가진 1차원임을 알수 있음
print(index[:5].values)
print(index.values[:5])
print(index[6])

# seris 형태로 인덱스하기

series_fair = titan_df['Fare']
print('Fair 컬럼의 MAX값 : ', series_fair.max())
print('Fair 컬럼의 SUM값 : ', series_fair.sum())
print('Fair 시리즈의 sum()값 : ', sum(series_fair))
print('Fair Series + 3 : \n', (series_fair + 3).head(3))

# DF나 Series에 .reset_index()메서드 수행시에 index 컬럼이 추가됨을 볼수 있음
titan_df_reset = titan_df.reset_index()
titan_df_reset.head(3)

print('### before reset_index ###')
value_counts = titan_df['Pclass'].value_counts()
print(value_counts) # Pclass의 고유값이 식별자 인덱스 역할을 하고 있음을 볼수 있음
print('value_count의 변수 타입', type(value_counts))
new_value_counts = value_counts.reset_index(inplace=False) # 원본데이터 값에는 변화가 없음
print('### After reset_index ###')
print(new_value_counts)
print('new_values_counts의 객체 변수 타입은 ? :', type(new_value_counts)) # series에서 데이터 프레임으로 변화함을 알 수 있다

# 판다스의 인덱스 및 셀렉션

# 판다스에는 위치,명칭 두개 모두 사용할수있는 ix[],명칭기반 인덱신 연산자 loc[], 칼럼 위치기반 인덱싱 iloc[]가 있다.
# ix 가 없어짐..

data = {'Name':['Wonsik','hyemin','jeawhan','sonia'],
        'Year':[1992,1997,1992,2000],
        'Gender':["Male","Female",'Male','Female']}
data_df = pd.DataFrame(data, index=['one','two','three','four'])
data_df
# data_df에서  reset_index() 새로운 숫자형 인덱스를 생성
data_df_reset = data_df.reset_index() # 0부터 초기화 되는 인덱스 생성
data_df_reset = data_df_reset.rename(columns={'index':'old_index'}) # 컬럼 네임을 올드 인덱스로 수젇ㅇ
# 인덱스 값에 1을 더해서 1부터 시작하는 새로운 인덱스 값 생성
data_df_reset.index = data_df_reset.index+1
data_df_reset

# iloc연산자 위치기반 인덱싱만 허용 행과 열값으로 정수형 or 정수형의 슬리이싱, 팬시리스트 값만 받을 수 있음
data_df_reset.iloc[0,1]
data_df_reset['one','name'] # 오류가 나는것을 확인 할수 있다.
# .loc연산자를 이용하여 명칭기반으로 인덱싱 하기
# .loc 지만 인덱싱 번호로도 호출이 가능하다
data_df.loc['one','Name']
data_df_reset.loc[1,'Name']
data_df_reset.loc[0,'Name'] # 0이라는 인덱싱이 존재하지 않기 때문에 오류가 발생한다.

print('명칭기반 ix 슬라이싱 \n', data_df.ix['one':'two','Name']) # 2022 1월 확인결과 ix 명령어 삭제됨
print('위치기반 iloc 슬라이싱 \n', data_df.iloc[0:1,0],'\n') # 1행 1열 출력
print('명칭기반 loc 슬라이싱 \n', data_df.loc['one':'two','Name']) # loc에서는 iloc에서와는 다르게 1행과2열에 행하는 인덱스의 값 두개를 반환한다.

# 불린 인덱싱
# 조건에 맞게 불린 인덱싱하여 추출하기
con1 = titan_df['Pclass'] == 1 # Pclass에서 1인 값만 추출 (불린형태로 볼수있음)
con2 = titan_df['Sex'] == 'female' # 성별이 여자인 사람만 추출
titan_df[con1 & con2] # 두개의 조건을 모두 만족하는 행을 인덱싱

# Sort 이용하여 정렬하기
titan_df_sort = titan_df.sort_values(by=['Name'])
titan_df_sort.head()
# 여러개의 칼럼으로 정렬하기
titan_df_sort_many = titan_df.sort_values(by=['Pclass','Name'],ascending=False) # pclass와 이름을 내림차순으로 정렬
titan_df_sort_many.head()
# Aggregation 함수를 적용하기 (min(),max(),sum(),count())
titan_df.count() # 각 컬럼의 카운트 결과를 반환
titan_df[['Age','Fare']].mean() # Age와 Fare 컬럼의 평균을 반환

#grouby 를 이용하여 하나로 묶기

titan_grouby = titan_df.groupby(by='Pclass')
print(type(titan_grouby)) # 데이터프레임 그룹바이라는 새로운 형태로 된것을 볼수있음.

titan_grouby = titan_df.groupby(by='Pclass').count()
titan_grouby # Pclass가 인덱스로 나머지 데이터를 묶은 것을 확인 할수 있다, grouby 대상을 제외한 모든 컬럼에 count가 적용이 됨.

# 두개의 칼럼을 필터링해 묶어보기
titan_grouby = titan_df.groupby(by='Pclass')[['PassengerId','Survived']].count() # Pcalss를 기준으로 패신져번호,생존 의 숫자를 카운트
titan_grouby

# Pclass 를 기준으로 나이를 최대, 최소값으로 구분하기
titan_grouby = titan_df.groupby(by='Pclass')['Age'].agg([min,max])
titan_grouby

#딕셔너리형태로 aggregation 함수 입력해서 사용하기
# SQL에서는 하나씩 편하게 사용이 가능 판다스에서는 딕셔너리형태로 변환하여 사용할 수 있다.
agg_format={'Age':'max','SibSp':'sum','Fare':'mean'}
titan_df.groupby('Pclass').agg(agg_format)

# NA 처리
# 결손값이 아닌것은 0 결손값인 것은 1로 표현 되기때문에 모두 더하면 결손값의 데이터 개수를 알 수 있다.
titan_df.isna().sum() # 카운트로하게되면 모든 값을 셈
titan_df['Cabin'] = titan_df['Cabin'].fillna('C000')
titan_df.head(3) # 결손값이 C000로 채워진걸 볼 수 있다.

titan_df['Age'] = titan_df['Age'].fillna(titan_df['Age'].mean()) # 타이타닉 데이터의 Age 칼럼의 결손치를 평균값으로 대체
titan_df['Embarked'] = titan_df['Embarked'].fillna('S')
titan_df.isna().sum()

# lambda 로 가공하기
def get_squre(x) :
    return x**2
print('5의 제곱은 : ' , get_squre(5) ) # 제곱을 구하는 사용자 정의함수 이다.

# 위의 사용자 정의 함수를 람다식으로 가공하기
lambda_squre = lambda x : x**2
print(' 5의 제곱은 : ', lambda_squre(5))
# 람다식에 여러개의 인자를 사용하고 싶을 경우에는 map함수를 이용
a = [1, 2, 3]
squares = map(lambda x : x**2, a)
list(squares)

titan_df['Name_len'] = titan_df['Name'].apply(lambda x : len(x))
titan_df[['Name','Name_len']].head(3) # 이름과 이름의 길이가 같이 출력된 것을 볼 수 있다.

# 나이를 이용하여 15세 미만이면 어린이 그렇지 않으면 어른인 컬럼을 생성하여 보여줌.
# 보통의경우 if절 뒤에 반환값이 오는데 람다의 경우는 if절 앞에 조건이 오는 것을 볼 수 있음.
titan_df['Child_Adult'] = titan_df['Age'].apply(lambda x : 'Child' if x <=15 else 'Adult')
titan_df[['Age','Child_Adult']].head(10)
# 나이를 이용하여 3단계 구분하기
# 15세 이하이면 어린이 60세 이하면 성인 나머지는 노년으로 데이터를 세분화
titan_df['Age_3'] = titan_df['Age'].apply(lambda x : 'Child' if x <= 15 else ('Adult' if x <= 60 else 'Elderly'))
titan_df[['Age','Age_3']].head(10)
titan_df['Age_3'].value_counts()

# 나이에 따라서 세분화 하는 함수 만들기
def get_category(age) :
    cat = ''
    if age <= 5 : cat = 'Baby'
    elif age <= 12 : cat = "Child"
    elif age <= 18 : cat = "Teenager"
    elif age <= 25 : cat = "Student"
    elif age <= 35 : cat = "Young Adult"
    elif age <= 60 : cat = "Adult"
    else : cat = 'Elderly'

    return cat
# 하나씩 받아서 수행
titan_df['Age_cat'] =  titan_df['Age'].apply(lambda x : get_category(x))
titan_df[['Age','Age_cat']].head
