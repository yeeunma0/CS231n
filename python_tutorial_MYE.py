#NOTE 기본자료형
###숫자
x=3
print (type(x)) #출력 "<type 'int'>"
print (x) #출력 "3"
print (x+1) #덧셈; 출력 "4"
print (x-1) #뺄셈; 출력 "2"
print (x*2) #곱셈; 출력 "6"
print (x**2) #제곱; 출력 "9"
x += 1
print (x) #출력 "4"
x *= 2
print (x) #출력 "8"
y=2.5
print (type(y)) #출력 <type 'float'>
print (y, y+1, y*2, y**2) #출력 "2.5, 3.5, 5.0, 6.25"
########NOTE 불리언
t=True
f=False
print (type(t)) #출력 "<type 'bool'>"
print (t and f) #논리 AND; 출력 "False"
print (t or f) #논리 OR; 출력 "True"
print (not t) #논리 NOT; 출력 "False"
print (t != f) #논리 XOR; 출력 "True"
###NOTE 문자열
hello = 'hello' #String 문자열을 표현할 땐 따옴표나
world = "world" #쌍따옴표가 사용됩니다; 어떤 걸 써도 상관없습니다.
print (hello) #출력 "hello"
print (len(hello)) #문자열 길이; 출력 "5"
hw = hello + ' ' + world #문자열 연결
print (hw) #출력 "hello world"
hw12 = '%s %s %d' % (hello, world, 12) #sprintf 방식의 문자열 서식 지정
print (hw12) #출력 "hello world 12"
s = "hello"
print (s.capitalize()) #문자열을 대문자로 시작하게 함; 출력 "Hello"
print (s.upper()) #모든 문자를 대문자로 바꿈; 출력 "HELLO"
print (s.rjust(7)) #문자열 오른쪽 정렬, 빈공간은 여백으로 채움; 출력 "  hello"
print (s.center(7)) #문자열 가운데 정렬, 빈공간은 여백으로 채움; 출력 " hello "
print (s.replace('l','(ell)')) #첫 번째 인자로 온 문자열을 두 번째 인자 문자열로 바꿈;
                             #출력 "he(ell)(ell)o"
print ('  world '.strip()) #문자열 앞뒤 공백 제거; 출력 "world"

#NOTE 컨테이너
##NOTE 리스트
xs = [3,1,2] #리스트 생성
print (xs, xs[2]) #출력 "[3,1,2]2"
print (xs[-1]) #인덱스가 음수일 경우 리스트의 끝에서부터 세어짐; 출력 "2"
xs[2] = 'foo' #리스트는 자료형이 다른 요소들을 저장할 수 있습니다
print (xs) #출력 "[3,1,'foo']"
xs.append('bar') #리스트의 끝에 새 요소 추가
print (xs) #출력 "[3,1,'foo','bar']"
x = xs.pop() #리스트의 마지막 요소 삭제하고 반환
print (x, xs) #출력 "bar [3,1,'foo']"
###슬라이싱
nums = range(5) #range는 파이썬에 구현되어 있는 함수이며 정수들로 구성된 리스트를 만듭니다
print (nums) #출력"[0,1,2,3,4]"
print (nums[2:4]) #인덱스 2에서 4(제외)까지 슬라이싱; 출력"[2,3]"
print (nums[2:]) #인덱스 2에서 끝까지 슬라이싱; 출력 "[2,3,4]"
print (nums[:2]) #인덱스 2에서 끝까지 슬라이싱; 출력 "[2,3,4]"
print (nums[:]) #전체 리스트 슬라이싱; 출력 "[0,1,2,3,4]"
print (nums[:-1]) #슬라이싱 인덱스는 음수도 가능; 출력 "[0,1,2,3]"
nums[2:4]=[8,9] #슬라이스된 리스트에 새로운 리스트 할당
print (nums) #출력 "[0,1,8,9,4]"
###반복문
animals = ['cat','dog','monkey']
for animal in animals:
    print (animal) #출력 "cat","dog","monkey", 한 줄에 하나씩 출력
for idx, animal in enumerate(animals):
    print ('#%d: %s') %(idx+1, animal)
#출력 "#1:cat", "#2:dog", "#3:monkey", 한 줄에 하나씩 출력
nums = [0,1,2,3,4]
squares = []
for x in nums:
    squares.append(x**2)
print (squares) #출력 [0,1,4,9,16]
squares = [x**2 for x in nums]
print (squares) #출력 [0,1,4,9,16]
even_squares = [x**2 for x in nums if x%2 == 0]
print (even_squares) #출력 "[0,4,16]"
##NOTE 딕셔너리
d={'cat':'cute','dog':'furry'} #새로운 딕셔너리를 만듭니다
print (d['cat']) #딕셔너리의 값을 받음; 출력 "cute"
print ('cat') in d #딕셔너리가 주어진 열쇠를 가졌는지 확인; 출력 "True"
d['fish'] = 'wet' #딕셔너리의 값을 지정
print (d['fish']) #출력 "wet"
#print s['monkey'] #KeyError : 'monkey' not a key of d
print (d.get('monkey','N/A')) #딕셔너리의 값을 받음. 존재하지 않는다면 'N/A'; 출력 "N/A"
print (d.get('fish','N/A')) #딕셔너리의 값을 받음. 존재하지 않는다면 'N/A'; 출력 "wet"
del d['fish'] #딕셔너리에 저장된 요소 삭제
print (d.get('fish','N/A')) #"fish"는 더 이상 열쇠가 아님; 출력 "N/A"
###반복문
d={'person':2, 'cat':4, 'spider':8}
for animal in d:
    legs = d[animal]
    print ('A %s has %d legs') %(animal, legs)
#출력 "A person has 2 legs", "A spider has 8 legs", "A cat has 4 legs", 한 줄에 하나씩 출력
d={'person':2,'cat':4,'spider':8}
for animal, legs in d.iteritems():
    print ('A %s has %d legs') %(animal, legs)
#출력 "A person has 2 legs", "A spider has 8 legs", "A cat has 4 legs", 한 줄에 하나씩 출력
nums = [0,1,2,3,4]
even_num_to_square = {x:x**2 for x in nums if x%2==0}
print (even_num_to_square) #출력 "{0:0, 2:4, 4:16}"
##NOTE 집합
animals={'cat','dog'}
print ('cat') in animals #요소가 집합에 포함되어 있는지 확인; 출력 "True"
print ('fish') in animals #출력 "False"
animals.add('fish') #요소를 집합에 추가
print ('fish') in animals #출력 "True"
print (len(animals)) #집합에 포함된 요소의 수; 출력 "3"
animals.add('cat') #이미 포함되어있는 요소를 추가할 경우 아무 변화 없음
print (len(animals)) #출력 "3"
animals.remove('cat') #Remove an element from a set
print (len(animals)) #출력 "2"
###반복문
animals = {'cat','dog','fish'}
for idx, animal in enumerate(animals):
    print ('#%d: %s') %(idx+1, animal)
#출력 "#1:fish", "#2:dog", "#3:cat", 한 줄에 하나씩 출력
from math import sqrt
nums = {int(sqrt(x)) for x in range(30)}
print (nums) #출력 "set([0,1,2,3,4,5])"
##NOTE 튜플
d={(x,x+1):x for x in range(10)} #튜플을 열쇠로 하는 딕셔너리 생성
t=(5,6) #튜플 생성
print (type(t)) #출력 "<type 'tuple'>"
print (d[t]) #출력 "5"
print (d[(1,2)]) #출력 "1"

#NOTE 함수
def sign(x):
    if x>0:
        return 'positive'
    elif x<0:
        return 'negative'
    else:
        return 'zero'
    
for x in [-1,0,1]:
    print (sign(x)) #출력 "negative","zero","positive", 한 줄에 하나씩 출력

def hello(name, loud=False):
    if loud : 
        print ('HELLO, %s!') %name.upper()
    else:
        print ('Hello, %s') %name

hello('Bob') #출력 "Hello, Bob"
hello('Fred',loud=True) #출력 "HELLO, FRED!"

#NOTE 클래스
class Greeter(object):

    #생성자
    def __init__(self, name) : 
        self.name = name #인스턴스 변수 선언

    #인스턴스 매소드
    def greet(self, loud=False):
        if loud :
            print ('HELLO, %s!') %self.name.upper()
        else :
            print ('Hello, %s') %self.name

g= Greeter('Fred') #Greeter클래스의 인스턴스 생성
g.greet() #인스턴스 매소드 호출; 출력 "Hello, Fred"
g.greet(loud=True) #인스턴스 메소드 호출; 출력 "HELLO,FRED!"