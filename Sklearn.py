import sklearn
import pandas as pd
from sklearn.datasets import load_iris  # 붓꽃의 데이터를 로드
from sklearn.tree import DecisionTreeClassifier  # 의사결정나무 임포트
from sklearn.model_selection import train_test_split  # 트레이닝셋과 테스트셋 스프릿
from sklearn.metrics import accuracy_score  # 정확도 확인하는 매서드
from sklearn.model_selection import StratifiedKFold # startifiedKfold 임포트
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

iris = load_iris()  # 아이리스 데이터셋 로드
# iris.data 는 피처만으로 된 데이터를 numpy로 가지고 있음.
iris_data = iris.data
# iris.target은 붓꽃 데이터 세트에서 레이블(결정값) 데이터를 numpy로 가지고 있음
iris_label = iris.target
print('iris target값 : ', iris_label)
print('iris target명 : ', iris.target_names)

# target = 품종(0,1,2 로 구분) , feature_names = 꽃잎크기,넓이등
# 붓꽃 데이터를 자세히 보기 위해서는 DataFrame으로의 변환이 필요하다.
iris_df = pd.DataFrame(data=iris_data, columns=iris.feature_names)
iris_df['label'] = iris.target  # 라벨이라는 새로운 컬럼 추가
iris_df.head(3)

# 트레이닝 셋과, 테스트셋 구분하기
# 테스트셋 20%, random_state = 시드설정 (난수발생을 일정화 시켜서 비교하기위해서)
X_train, X_test, y_train, y_test = train_test_split(iris_data, iris_label,
                                                    test_size=0.2, random_state=1)
# 판다스로도 데이터셋 스플릿이 가능하다 .
import pandas as pd

iris_data = pd.DataFrame(data=iris_data.data, columns=iris_data.feature_names)
iris['target'] = iris_data.target
iris_df.head(3)
# 의사 결정 나무 생성
dt = DecisionTreeClassifier(random_state=1)
# 학습 시키기
dt.fit(X_train, y_train)
# 예측결과보기
pred = dt.predict(X_test)
# 예측정확도 확인하기
print('예측 정확도 : {0:.3f}'.format(accuracy_score(y_test, pred)))  # 예측정확도가 96.7%가 나온것을 알 수 있다.

# 사이킷런은 머신러닝 모델학습을 위하여 .fit을 사용, 모델의 예측을 위해서는 .predict를 사용
# 사이킷런은 분류를 위한 알고리즘 클래스를 Classifier 회귀를 위한 알고리즘 클래스를 Regressor클래스라고 칭함
# 분류와 회귀를 위한 알고리즘 클래스를 모두 합쳐 Estimator라고 칭함 (부모클라스라고 생각하면 편하다)

from sklearn.datasets import load_iris

iris_data = load_iris()
print(type(iris_data))  # Bunch클래스는 Python의 딕셔너리 클래스와 유사하다

# iris 데이터의 키값들 알아보기
keys = iris_data.keys()
print('iris 데이터셋의 키값들 : ', keys)

# iris 데이터의 나온 키값을 하나씩 조회해보기
# feature_names 의 데이터 조회
print('\n feature_names의 타입 : ', type(iris_data.feature_names))  # 리스트 형태로 되어있으며
print('feature_names 의 형태 : ', len(iris_data.feature_names))  # 4개의 값을 가지고 있다.
print(iris_data.feature_names)
# target_names의 데이터 조회
print('\n target_names의 타입 : ', type(iris_data.target_names))  # ndarray 형태로 되어있으며
print('target_names의 형태 : ', len(iris_data.target_names))  # 3개의 값을 가지고 있다.
print(iris_data.target_names)
# data의 데이터 조회
print('\n data의 타입 : ', type(iris_data.data))  # ndarray의 형태를 가지고있으며
print('data의 형태 : ', len(iris_data.data))  # 150개의 데이터를 가지고 있다.
print(iris_data.data)
# target의 데이터 조회
print('\n target의 타입 : ', type(iris_data.target))  # ndarray의 형태를 가지고있으며
print('target의 형태 : ', len(iris_data.target))  # 150개의 데이터를 가지고 있다.
print(iris_data.target)

# 판다스로 데이터 스플릿 셋 만들기
import pandas as pd

iris_data = load_iris()
iris_df = pd.DataFrame(data=iris_data.data, columns=iris_data.feature_names)
iris_df['target'] = iris_data.target
iris_df.head(3)
# feature 데이터프레임,  target데이터 프레임 생성하기
ftr_df = iris_df.iloc[:, :-1]
target_df = iris_df.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(ftr_df, target_df, test_size=0.3, random_state=1)

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
import numpy as np

iris = load_iris()
features = iris.data
label = iris.target
iris_label = iris.target
df = DecisionTreeClassifier(random_state=1)

# 5개의 폴드 세트로 분리하는 Kfold 객체와 폴드 세트별 정확도를 담을 리스트 객체를 생성
kfold = KFold(n_splits=5)
cv_accuracy = []
print('붓꽃 데이터 세트의 크기 : ', features.shape[0])

# KFold 객체의 split()를 호출하면 폴드 별 학습용, 검증용 테스트의 로우 인덱스를 array로 변환
counter = 0
for train_index, test_index, in kfold.split(features):
    X_train, X_test = features[train_index], features[test_index]
    y_train, y_test = label[train_index], label[test_index]
    # 학습,예측하기
    df.fit(X_train,y_train)
    pred = df.predict(X_test)
    counter +=1
    # 반복할때마 정확도를 측정함
    accuracy = np.round(accuracy_score(y_test,pred),4)
    train_size = X_train.shape[0]
    test_size = X_test.shape[0]
    print('\n#{0} 교차 검증 정확도 : {1}' '학습 데이터 크기 : {2}' '검증 데이터 크기 : {3}'.format(counter,accuracy,train_size,test_size))
    print('#{0} 검증 세트 인덱스 : {1}'.format(counter,test_index))
    cv_accuracy.append(accuracy)

print('\n## 평균 검증 정확도 : ', np.mean(cv_accuracy))

# startified K폴드 검정

import pandas as pd

iris = load_iris()
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df['label'] = iris.target #라벨이라는 컬럼 생성하여 타켓을 넣어줌
iris_df['label'].value_counts() # 0,1,2 가 각각 50개 씩 들어간걸 볼수있음


## 이렇게만들면 3개로 나눴을때 하나의 조건이 아예 배제되는 현상이 발생한다.
kfold = KFold(n_splits=3) #3개의 데이터셋으로 분할
counter = 0
for train_index,test_index in kfold.split(iris_df):
    counter +=1
    label_train = iris_df['label'].iloc[train_index]
    label_test = iris_df['label'].iloc[test_index]
    print('교차검증 {0}'.format(counter))
    print('학습데이터 분포 : \n ', label_train.value_counts())
    print('검증 레이블 데이터 분포 : \n' , label_test.value_counts())

# 때문에 StartifedKFold 방식으로 데이터를 처리 하여야 한다.
from sklearn.model_selection import StratifiedKFold # startifiedKfold 임포트

stk = StratifiedKFold(n_splits=3)
counter = 0

for train_index,test_index in stk.split(iris_df,iris_df['label']):
    counter += 1
    label_train = iris_df['label'].iloc[train_index]
    label_test = iris_df['label'].iloc[test_index]
    print('## 교차 검증 : {0}'.format(counter))
    print('학습 레이블 데이터 분포 : \n', label_train.value_counts())
    print('테스트 레이블 데이터 분포 : \n', label_test.value_counts())

# StartifiedKFold로 분배된 데이터로 의사결정나무 학습하기

stk_dt = DecisionTreeClassifier(random_state=1)

stk = StratifiedKFold(n_splits=3) # StartifedKfold 3분할
counter = 0  # 3개의 데이터가있으니 3번 반복함
accuracy_ls = [] #정확도를 담을 list

for train_index,test_index in stk.split(features, label):
    X_train, X_test = features[train_index], features[test_index]
    y_train, y_test = label[train_index], label[test_index]
    stk_dt.fit(X_train,y_train) # 학습용 세트 학습
    predict = stk_dt.predict(X_test) # 테스트 셋으로 예측

    counter += 1
    accuracy = np.around(accuracy_score(y_test,predict),4) #소수점 4자리까지 반올림
    train_size = X_train.shape[0]
    test_size = X_test.shape[0]
    print('\n{0} 교차 검증 정확도 :{1} 학습데이터 크기 : {2}' '검증 데이터 크기 : {3}'.format(counter,accuracy,train_size,test_size))
    print('#{0} 검증 세트 인덱스 :{1}'.format(counter,test_index))
    accuracy_ls.append(accuracy)
    print('\n 교차 검증별 정확도 : ', np.around(accuracy_ls,4))
    print('평균 검증 정확도 : ', np.mean(accuracy_ls))

# 사이킷런에서는 편하게 교차검증을 할수 있는 API를 제공한다.
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score,cross_validate
from sklearn.datasets import load_iris

iris_data = load_iris()
dt_iris = DecisionTreeClassifier(random_state=1)

data = iris_data.data
label = iris_data.target
# 성능지표를 설정할수있는데 성능 지표는 accuracy로 지정한다.
score = cross_val_score(dt_iris, data, label, scoring='accuracy', cv=3) # 정확도 를 성능지표로 3분할
print('교차 검증별 정확도 : ', np.around(score,4))
print('평균 검증 정확도 : ', np.around(np.mean(score),4))

# GridsearchCV 하이퍼 파라미터를 튜닝해서 최적화된 알고리즘 찾기
from sklearn.model_selection import  GridSearchCV
import pandas as pd
iris_data = load_iris()
X_train,X_test,y_train,y_test = train_test_split(iris_data.data, iris_data.target, test_size=0.2, random_state=1)

dt_iris = DecisionTreeClassifier()

parameters = {'max_depth':[1,2,3],'min_samples_split':[2,3]}

gird_dtree = GridSearchCV(dt_iris,param_grid=parameters, cv=3, refit=True)
# 붓꽃 학습 데이터로 param_grid의 하이퍼 파라미터를 순차적으로 학습/평가
gird_dtree.fit(X_train, y_train)
# GridSearchCV 결과를 추출해 DatgaFrame으로 변환
score_df = pd.DataFrame(gird_dtree.cv_results_)
score_df[['params','mean_test_score','rank_test_score', 'split0_test_score','split1_test_score','split2_test_score']]
# GridsearchCV를 통해 추출한 결과에서 필요한 자료만 뽑기

print('GridserchCV 최적의 파라미터 : ', gird_dtree.best_params_)
print('GridserchCV 최고 정확도 : {0:0.4f} '.format(gird_dtree.best_score_))

# 최고 측정치로 학습된 결과는 best_estimator_에 저장이 됨. 이미 최적으로 학습된 상태이므로 학습을 해줄 필요가 없다.
estimator = gird_dtree.best_estimator_
predict = estimator.predict(X_test)
print('테스트 데이터 테스트 정확도 : {0:.4f}'.format(accuracy_score(y_test,predict)))
