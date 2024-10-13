import numpy as np

class NearestNeighbor(object):
    def __init__(self):
        pass

    def train(self,X,y):
        """X is NxD where each row is an example. Y is 1-dimension of size N"""
        # nearest neighbor 분류기는 단순히 모든 학습 데이터를 기억해둔다.
        self.Xtr = X
        self.Ytr = y

    def predict(self,X):
        """X is NxD where each row is an example we wish to predict label for"""
        num_test = X.shape[0]
        #출력 type과 입력 type이 갖게 되도록 확인해준다.
        Ypred = np.zeros(num_Test,dtype = self.ytr.dtype)

        #loop over all test row
        ##파이썬 구버전에서 사용하던 xrange대신 최신버전에서는 range사용함.
        for i in range(num_test):
            #i번째 테스트 이미지와 가장 가까운 학습 이미지를
            #L1 거리(절댓값 차의 총합)를 이용하여 찾는다.
            distances = np.sum(np.abs(self.Xtr - X[i,:]),axis = 1)
            min_index = np.argmin(distances) #가장 작은 distance를 갖는 인덱스를 찾는다.
            Ypred[i] = self.ytr[min_index] #가장 가까운 이웃의 라벨로 예측

        return Ypred