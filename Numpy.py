import numpy as np

# ndarray를 사용하여 배열 만들기
array1 = np.array([1,2,3])
print('array 1의 타입은:', type(array1))

array2 = np.array([[1,2,3],
                   [2,3,4]])
print('array 2의 타입은:', type(array2))

array3 = np.array([[1,2,3]])
print('array 3의 타입은:', type(array3))
print('arry 3의 형태:', array3.shape) # array 1과는 다르게 2차원 형태의 배열을 가진것을 볼수있다.

# ndim을 이용하여 차원수를 알수있다.
print('array1: {:0}차원, array2:{:1}차원, array3:{:2}차원'.format(array1.ndim,array2.ndim,array3.ndim))

# ndarray의 데이터값은 int,float,chr,bull 모두 가능하다.
# 데이터값이 불일치할경우 데이터의 형태가 변할수 있으니 유의 해야한다.
list = [1,2,"char"]
array_ch = np.array(list)
print(array_ch, array_ch.dtype)
# 모두 문자형으로 변한것을 볼수있다 .

list2 = [1,2,3.5]
array_fl = np.array(list2)
print(array_fl,array_fl.dtype)
# 모두 실수형으로 변한것을 볼수있다.

# astype을 이용하여 형 변환하기
array_int = np.array([1,2,3,4])
array_float = array_int.astype('float64')
print(array_float, array_float.dtype)
# int 형의 리스트가 float 형으로 변한 것을 볼수 있다.

# 특수한  ndarray  생성하기

# arange 를 이용하면 range와 같이 순차적으로 생성이 가능하다.
seq_array = np.arange(10)
print(seq_array)
print(seq_array.dtype,seq_array.shape)

# zeros 이용하여 0이들어간 ndarray 생성하기
zero_array = np.zeros((3,2), dtype="int32") #int32형으로 3행 2형로 생성
print(zero_array)
print(zero_array.dtype,zero_array.shape)

# ones를 이용하면 1이 들어간 ndarray를 생성할수있다.
# dtype을 정하지않고 디폴트로 줄경우 float 64형으로 출력
one_array = np.ones((3,2))
print(one_array)
print(one_array.dtype,one_array.shape)

# reshape를 통해 배열의 차원과 크기를 재배열 할수있다.
array1 = np.arange(20)
print('array1 : ',array1) #(1~20의 연속적 배열 생성)
array2 = array1.reshape(4,5)
print('array2 : ', array2) #(4행 5열의 행렬로 재배치)

# reshape에 -1을 사용하여 사이즈 를 변경하기
array3 = array1.reshape(-1,5) # 20개를 5열로 배치하기위해 4행이 필요하다. -1을 이용하면 이를 찾아 빠르게 사이즈를 변경 할 수 있다.
print('array3 : ',array3)

# reshape를 이용하여 차원 변경하기
array1 = np.arange(8)
array_3d = array1.reshape(2,2,2)
print('array_3d :', array_3d.tolist())

#3차원에서 2차원으롭 변경하기
array_2d = array_3d.reshape(-1,1)
print('array_2d : ', array_2d.tolist()) # tolist를 이용하면 리스트 형태로 볼 수 있음
print('array_2d shape : ', array_2d.shape)

# 인덱싱
# 넘버링으로 인덱싱하기
array1 = np.arange(start=1, stop=10) # 1~9의 값을 가진 리스트 생성
print(array1[2]) # 3번째 있는 값을 출
# 2차원 행렬 넘버링으로 인덱싱
array2 = array1.reshape(3,3)
print(array2)
print(array2[1,2]) # 2행 3번째 값을 가르킴 6을 출력

# 슬라이싱을 이용하여 인덱싱하기
# 슬라이싱 기호인 ":"를 이용하여 인덱싱 할수있다.
array_sl = np.arange(start=1, stop=10)
array_sl1 = array_sl[:3] # 기호앞의 숫자를 생락 하면 0으로 간주한다.
print(array_sl1)
array_sl2 = array_sl[3:] # 기호뒤의 숫자를 생략하면 마지막 인덱스번호로 간주한다.
print(array_sl2)
array_sl3 = array_sl[:] # 기호 앞뒤 숫자를 생략하면 맨처음 맨마지막 인덱스로간주 (전부다 출력)
print(array_sl3)
# 펜시 인덱싱
array_1d = np.arange(start=1,stop=10)
array_2d = array_1d.reshape(3,3)

array_f = array_2d[[0,1],2] # 1행과2행의 3열을 표시
print('array_f[[0,1],2] = ', array_f.tolist())

array_f2 = array_2d[[0,1],0:2] #1행과2행의 1열과 2열을 출력
print('array_2d[[0,1],0:2] = ' , array_f2.tolist())

# 불린 인덱싱 (많이 사용됨)
array_bo = array_1d[array_1d > 5]
print('array_1d에서 5보다 큰값 = ', array_bo)
array_1d > 5
bool_idx = np.array([False, False, False, False, False,  True,  True,  True,  True])
array_bool = array_1d[bool_idx]
print("불린 인덱스로 값내기", array_bool)

# 행렬 정렬하기 sort
dis_array = np.array([4,2,1,5])
print('원래 행렬값',dis_array)
sort_array = np.sort(dis_array)
print('정렬 후의 행렬 값', sort_array)
print('정렬후의 원래 행렬값', dis_array)
sort_array2 = dis_array.sort()
print('원래 행렬값 정렬후 반환된 행렬', sort_array2)
print('원본 행렬',dis_array)
# np.sort와 ndarray.sort의 차이이다
# np.sort의 경우는 원래의 행렬값을 변경하지 않고
# ndarray.sort 는 원래 행렬값을 변경 하는 것을 볼 수 있다.

sort_array_des = np.sort(dis_array)[::-1]
print(sort_array_des)
# 내림차순으로 전렬 하고 싶을때는 np.sort()[::-1]을 이용하면 된다.

# 원본행렬의 원소에대한 인덱스를 필요로 할때는 np.argsort()를 이용한다.
# 인덱스 내림차순의 경우는 np.argsort()[::-1]을 이용하면 된다.
dis_array = np.array([4,3,5,8])
sort_idx = np.argsort(dis_array)
print(type(sort_idx))
print('행렬을 정렬했을때 원본의 인덱스의 값 : ', sort_idx)

# 키와 밸류의값이 암묵적으로 정해져 있을때 활용도가 높다.

name_array = np.array(['스파이더맨','아이언맨','헐크','블랙위도우','닥터스트레인지'])
power_array = np.array([350,700,950,500,1000])

sort_idx_pwr = np.argsort(power_array)
print('제일 강한 마블 히어로 순서 : ', name_array[sort_idx_pwr])

# 선형대수 계산

# 행렬의 곱은 np.dot을 이용하여 구할수있다.
array1 = np.array([[1,2,3],[4,5,6]])
array2 = np.array([[7,8],[9,10],[11,12]])
np_dot = np.dot(array1,array2)
print('행렬곱 = ', np_dot)

# 전치 행렬
# 전치 행렬은 행과 열의 위치를 교환한 원소로 구성한 행렬이다.

matrix = np.array([[1,2],
                   [3,4]])
trans_mtx = np.transpose(matrix)
print('1,2,3,4 2*2의 전치행렬은 ', trans_mtx)
