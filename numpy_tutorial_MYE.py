"""Numpy 배열은 동일한 자료형을 가지는 값들이 격자판 형태로 있는 것
각각의 값들은 튜플의 형태로 색인
rank는 배열이 몇 차원인지를 의미
shape는 각 차원의 크기를 알려주는 정수들이 모인 튜플"""

# NOTE 배열
import numpy as np
a= np.array([1,2,3]) #rank가 1인 배열 생성
print (type(a)) #출력 "<type 'numpy.ndarray'>"
print (a.shape) #출력 "(3,)"
print (a[0], a[1], a[2]) #출력 "1,2,3"
a[0]=5 #요소를 변경
print (a) #출력 "[5,2,3]"

b=np.array([[1,2,3],[4,5,6]]) #rank가 2인 배열 생성
print (b.shape) #출력 "(2,3)" ##밖에 괄호부터 차원을 세주는 느낌이다?
print (b[0,0], b[0,1], b[1.0]) #출력 "1, 2, 4"

a = np.zeros((2,2)) #모든 값이 0인 배열 생성
print (a) #출력 "[[0.0.]
        #         [0.0.]]"

b = np.ones((1,2)) #모든 값이 1인 배열 생성
print (b) #출력 "[[1.1.]]"

c = np.full((2,2),7) #모든 값이 특정 상수인 배열 생성
print (c) #출력 "[[7.7.]
          #       [7.7.]]"

d = np.eye(2) #2x2 단위행렬 생성
print (d) #출력 "[[1.0.]
          #       [0.1.]]"

e = np.random.random((2,2)) #임의의 값을 채워진 배열 생성
print (e) #임의의 값 출력 "[[0.34241 0.421423]
          #                [0.34145 0.873424]]"

#NOTE 배열 인덱싱
##슬라이싱
#Create the following rank 2 array with shape (3,4)
#[[1  2  3  4]
# [5  6  7  8]
# [9 10 11 12]]
a = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]])

#슬라이싱을 이용하여 첫 두 행과 1열, 2열로 이루어진 부분배열을 만들어봅시다
#b는 shape가 (2,2)인 배열이 됩니다
#[[2 3]
#  [6 7]]
b = a[:2, 1:3]

#슬라이싱된 배열은 원본 배열과 같은 데이터를 참조합니다
#즉, 슬라이싱된 배열을 수정하면 원본 배열 역시 수정됩니다.
print (a[0,1]) #출력 "2"
b[0,0] = 77 #b[0,0]은 a[0,1]과 같은 데이터입니다
print (a[0,1]) #출력 "77"

#배열의 중간 행에 접근하는 두 가지 방법이 있습니다
#정수 인덱싱과 슬라이싱을 혼합해서 사용하면 낮은 rank의 배열이 생성되지만.
#슬라이싱만 사용하면 원본 배열과 동일한 rank의 배열이 생성됩니다.
row_r1 = a[1,:] #배열a의 두 번째 행을 rank가 1인 배열로
row_r2 = a[1:2,:] #배열a의 두 번째 행을 rank가 2인 배열로
print (row_r1, row_r1.shape) #출력 "[5 6 7 8](4,)"
print (row_r2, row_r2.shape) #출력 "[[5 6 7 8]] (1,4)"

#행이 아닌 열의 경우에도 마찬가지입니다:
col_r1 = a[:,1]
col_r2 = a[:,1:2]
print (col_r1,col_r1.shape) #출력 "[2 6 10](3,)"
print (col_r2,col_r2.shape) #출력 "[[2]
                            #       [6]
                            #       [10]](3,1)"
##정수 배열 인덱싱
a = np.array([[1,2],[3,4],[5,6]])

#정수 배열 인덱싱의 예
#반환되는 배열의 shape는 (3,)
print (a[[0,1,2],[0,1,0]]) #출력 "[1 4 5]"

#위에서 본 정수 배열 인덱싱 예제는 다음과 동일합니다:
print (np.array([a[0,0],a[1,1],a[2,0]])) #출력 "[1 4 5]"

#정수 배열 인덱싱을 사용할 때,
#원본 배열의 같은 요소를 재사용할 수 있습니다.

print (a[[0,0],[1,1]]) #출력 "[2 2]"

#위 예제는 다음과 동일합니다
print (np.array([a[0,1],a[0,1]])) #출력 "[2 2]"

#요소를 선택할 새로운 배열 생성
a = np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]])

print (a) #출력 "array([[1,2,3],
          #            [4,5,6],
          #            [7,8,9]
          #            [10,11,12]])"

#인덱스를 저장할 배열 생성
b = np.array([0,2,0,1])

#b에 저장된 인덱스를 이용해 각 행에서 하나의 요소를 선택합니다
print (a[np.arange(4),b]) #출력 "[1 6 7 11]"

#b에 저장된 인덱스를 이용해 각 행에서 하나의 요소를 변경합니다.
a[np.arange(4),b] += 10

print (a) #출력 "array([[11,2,3]
        #             [4,5,16]
        #             [17,8,9]
        #             [10,21,12]])"

##NOTE 불리언 배열 인덱싱
a = np.array([[1,2],[3,4],[5,6]])

bool_idx = (a>2) #2보다 큰 a의 요소를 찾습니다.
                 #이 코드는 a와 shape가 같고 불리언 자료형을 요소로 하는 numpy 배열을 반환합니다.
                 #bool_idx의 각 요소는 동일한 위치에 있는 a의 요소가 2보다 큰지를 말해줍니다.

print (bool_idx) #출력 "[[False False]
                 #       [True True]
                 #       [True True]]"

#불리언 배열 인덱싱을 통해 bool_idx에서 참값을 가지는 요소로 구성되는
#rank 1인 배열을 구성할 수 있습니다.
print (a[bool_idx]) #출력 "[3 4 5 6]"

#위에서 한 모든 것을 한 문장으로 할 수 있습니다:
print (a[a>2]) #출력 [3 4 5 6]

#NOTE 자료형
x = np.array([1,2]) #numpy가 자료형을 추측해서 선택
print (x.dtype) #출력 "int64"

x = np.array([1.0,2.0]) #numpy가 자료형을 추측해서 선택
print (x.dtype) #출력 "float64"

x = np.array([1,2],dtype=np.int64) #특정 자료형을 명시적으로 지정
print (x.dtype) #출력 "int64"

#NOTE 배열 연산
x = np.array([[1,2],[3,4]], dtype=np.float64)
y = np.array([[5,6],[7,8]], dtype=np.float64)

#요소별 합; 둘 다 다음의 배열을 만듭니다.
#[[6.0  8.0]
# [10.0 12.0]]
print (x+y)
print (np.add(x,y))

#요소별 차; 둘 다 다음의 배열을 만듭니다
#[[-4.0 -4.0]
# [-4.0 -4.0]]
print (x-y)
print (np.subtract(x,y))

#요소별 곱; 둘 다 다음의 배열을 만듭니다.
#[[5.0 12.0]
# [21.0 32.0]]
print (x*y)
print (np.multiply(x,y))

#요소별 나눗셈; 둘 다 다음의 배열을 만듭니다.
#[[0.2  0.3333]
# [0.4258  0.5]]
print (x/y)
print (np.divide(x,y))

#요소별 제곱근; 다음의 배열을 만듭니다
#[[1.      1.4142]
# [1.7320      2.]]
print (np.sqrt(x))

v = np.array([9,10])
w = np.array([11,12])

#벡터의 내적;둘 다 결과는 219
print (v.dot(w))
print (np.dot(v,w))

#행렬과 벡터의 곱; 둘 다 결과는 rank 1인 배열 [29 67]
print (x.dot(v))
print (np.dot(x,v))

#행렬곱; 둘 다 결과는 rank 2인 배열
#[[19 22]
# [43 50]]
print (x.dot(y))
print (np.dot(x,y))

print (np.sum(x)) #모든 요소를 합한 값을 연산; 출력 "10"
print (np.sum(x,axis=0)) #각 열에 대한 합을 연산; 출력 "[4 6]"
print (np.sum(x,axis=1)) #각 행에 대한 합을 연산; 출력 "[3 7]"

print (x) #출력 "[[1 2]
          #            [3 4]]"
print (x.T) #출력 "[[1 3]
            #            [2 4]]"

#rank 1인 배열을 전치할 경우 아무일도 일어나지 않습니다.
v = np.array([1,2,3])
print (v) #출력 "[1 2 3]"
print (v.T) #출력 "[1 2 3]"

#NOTE 브로드 캐스팅
#numpy에서 shape가 다른 배열 간에도 산술 연산이 가능하게 하는 메커니즘입니다
#행렬의 x의 각 행에 벡터 v를 더한 뒤,
#그 결과를 행렬 y에 저장하고자 합니다
x = np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]])
v = np.array([1,0,1])
y = np.empty_like(x) #x와 동일한 shape를 가지며 비어있는 행렬 생성

#명시적 반복문을 통해 행렬 x의 각 행에 벡터 v를 더하는 방법
for i in range(4):
    y[i,:]=x[i,:]+v

#이제 y는 다음과 같습니다
#[[2 2 4]
# [5 5 7]
# [8 8 10]
# [11 11 13]]
print (y)

#'x'가 매우 큰 행렬이라면,
#파이썬의 명시적 반복문을 이용한 위 코드는 매우 느려질 수 있습니다.
#벡터 'v'를 행렬 'x'의 각 행에 더하는 것은
#'v'를 여러 개 복사해 수직으로 쌓은 행렬 'vv'를 만들고
#이 'vv'를 'x'에 더하는 것과 동일합니다.
vv = np.tile(v,(4,1)) #v의 복사본 4개를 위로 차곡차곡 쌓은 것이 vv
print (vv) #출력 "[[1 0 1]
           #       [1 0 1]
           #       [1 0 1]
           #       [1 0 1]]"
y = x+vv #x와 vv의 요소별 합
print (y) #출력 "[[2 2 4]
          #       [5 5 7]
          #       [8 8 10]
          #       [11 11 13]]"

#numpy 브로드캐스팅을 이용한다면 이렇게 v의 복사본을 여러 개 만들지 않아도
#동일한 연산을 할 수 있습니다.
y=x+v #브로드캐스팅을 이용하여 v를 x의 각 항에 더하기
print (y) #출력 "[[2 2 4]
          #     [5 5 7]
          #     [8 8 10]
          #     [11 11 13]]"

#브로드캐스팅을 지원하는 함수를 universal functions라고 합니다
v = np.array([1,2,3]) #v의 shape는 (3,)
w = np.array([4,5]) #w의 shape는 (2,)
#외적을 계산하기 위해, 먼저 v를 shape가 (3,1)인 행벡터로 바꿔야합니다;
#그다음 이것을 w에 맞춰 브로드캐스팅한 뒤 결과물로 shape가 (3,2)인 행렬을 얻습니다.
#이 행렬은 v와 w의 외적의 결과입니다:
#[[4 5]
# [8 10]
# [12 15]]
print (np.reshape(v,(3,1))*w)

#벡터를 행렬의 각 행에 더하기
x = np.array([[1,2,3],[4,5,6]])
#x는 shape가 (2,3)이고 v는 shape가 (3,)이므로 이 둘을 브로드캐스팅하면 shape가 (2,3)인
#아래와 같은 행렬이 나옵니다:
#[[2 4 6]
# [5 7 9]]
print (x+v)

#벡터를 행렬의 각 행에 더하기
#x는 shape가 (2,3)이고 w는 shape가 (2,)입니다.
#x의 전치행렬은 shape가 (3,2)이며 이는 w와 브로드캐스팅이 가능하고
#결과로 shape가 (3,2)인 행렬이 생깁니다;
#이 행렬을 전치하면 shape가 (2,3)인 행렬이 나오며
#이는 행렬 x의 각 열에 벡터 w를 더한 결과와 동일합니다.
#아래의 행렬입니다:
#[[5 6 7]
# [9 10 11]]
print (x.T+w).T
#다른 방법은 w를 shape가 (2,1)인 열벡터로 변환하는 것입니다;
#그런 다음 이를 바로 브로드캐스팅해 더하면
#동일한 결과가 나옵니다.
print (x + np.reshape(w,(2,1)))
#행렬의 스칼라배:
#x의 shape는 (2,3)입니다. numpy는 스칼라는 shape가 ()인 배열로 취급합니다;
#그렇기에 스칼라 값은 (2,3) shape로 브로드캐스트 될 수 있고,
#아래와 같은 결과를 만들어 냅니다.
#[[2 4 6]
# [8 10 12]]
print (x*2)