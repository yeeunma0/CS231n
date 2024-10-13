#matplotlib는 plotting 라이브러리입니다.
 
#NOTE plotting
#matplotlib에서 가장 중요한 함수는 2차원 데이터를 그릴 수 있게 해주는 plot입니다.
import numpy as np
import matplotlib.pyplot as plt

#사인과 코사인 곡선의 x,y 좌표를 계산
x = np.arange(0,3*np.pi, 0.1)
y = np.sin(x)

#matplotlib를 이용해 점들을 그리기
plt.plot(x,y)
plt.show() #그래프를 나타나게 하기 위해선 plt.show()함수를 호출해야만 합니다.

#사인과 코사인 곡선의 x,y 좌표를 계산
x = np.arange(0,3*np.pi,0.1)
y_sin = np.sin(x)
y_cos = np.cos(x)

#matplotlib를 이용해 점들을 그리기
plt.plot(x,y_sin)
plt.plot(x,y_cos)
plt.xlabel('x axis label')
plt.ylabel('y axis label')
plt.title('Sine and Cosine')
plt.legend(['Sine','Cosine'])
plt.show()

#NOTE subplot
#'subplot'함수를 통해 다른 내용도 동일한 그림 위에 나타낼 수 있습니다.
import numpy as np
import matplotlib.pyplot as plt

#사인과 코사인 곡선의 x,y 좌표를 계산
x = np.arange(0,3*np.pi,0.1)
y_sin = np.sin(x)
y_cos = np.cos(x)

#높이가 2이고 너비가 1인 subplot 구획을 설정하고,
#첫 번째 구획을 활성화.
plt.subplot(2,1,1)#2행 1열의 구조 1번쨰 그래프

#첫 번째 그리기
plt.plot(x,y_sin)
plt.title('Sine')

#두 번째 subplot 구획을 활성화하고 그리기
plt.subplot(2,1,2)#2행 1열의 구조 2번째 그래프
plt.plot(x,y_cos)
plt.title('Cosine')

#그림 보이기
plt.show()

#NOTE 이미지
#imshow함수를 사용해 이미지를 나타낼 수 있습니다.

import numpy as np
from scipy.misc import imread, imresize
import matplotlib.pyplot as plt

img = imread('aseets/cat'.jpg)
img_tinted = img*[1,0.5,0.9]

#원본 이미지 나타내기
plt.subplot(1,2,1)#1행 2열의 구조 1번째 그래프
plt.imshow(img)

#색변화된 이미지 나타내기
plt.subplot(1,2,2)
#inshow를 이용하며 주의할 점은 데이터의 자료형이
#unit8이 아니라면 이상한 결과를 보여줄 수도 있다는 것입니다.
#그러므로 이미지를 나타내기 전에 명시적으로 자료형을 unit8로 형변환해줍니다.
plt.imshow(np.unit8(img_tinted))
plt.show()