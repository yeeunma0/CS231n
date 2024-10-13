#필요한 도구들을 임포트
import torch
import random
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.nn
import matplotlib.pyplot as plt

#만약 GPU 사용 가능하면 device 값이 cuda, 아니면 cpu
device = 'cuda' if torch.cuda.is_available() else 'cpu' 

# 랜덤 시드 고정
torch.manual_seed(777)

# GPU 사용 가능일 경우 랜덤 시드 고정
if device == 'cuda':
    torch.cuda.manual_seed_all(777)

#학습에 사용할 파라미터 설정
learning_rate = 0.001
training_epochs = 15
batch_size = 100

#데이터 전처리
mnist_train = dsets.MNIST(root='MNIST_data/', # 다운로드 경로 지정
                          train=True, # True를 지정하면 훈련 데이터로 다운로드
                          transform=transforms.ToTensor(), # 텐서로 변환
                          download=True)

mnist_test = dsets.MNIST(root='MNIST_data/', # 다운로드 경로 지정
                         train=False, # False를 지정하면 테스트 데이터로 다운로드
                         transform=transforms.ToTensor(), # 텐서로 변환
                         download=True)

data_loader = torch.utils.data.DataLoader(dataset=mnist_train,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          drop_last=True)

#합성곱층과 풀링층
class CNN(torch.nn.Module):

    def __init__(self):
        super(CNN, self).__init__()
        # 첫번째층
        # ImgIn shape=(?, 28, 28, 1) 여기서 ?는 배치사이즈
        #    Conv     -> (?, 28, 28, 32)
        #    Pool     -> (?, )14, 14, 32
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))
        #합성곱(nn.Cov2d)+활성화 함수(nn.ReLU)+맥스풀링(nn.MaxPool2d)을
        #하나의 합성곱 층으로 본다.

        # 두번째층
        # ImgIn shape=(?, 14, 14, 32)
        #    Conv      ->(?, 14, 14, 64)
        #    Pool      ->(?, 7, 7, 64)
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))

        # 전결합층 7x7x64 inputs -> 10 outputs
        self.fc = torch.nn.Linear(7 * 7 * 64, 10, bias=True)

        # 전결합층 한정으로 가중치 초기화
        torch.nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)   # 전결합층을 위해서 Flatten
        out = self.fc(out)
        return out

# CNN 모델 정의
model = CNN().to(device)

criterion = torch.nn.CrossEntropyLoss().to(device)    # 비용 함수에 소프트맥스 함수 포함되어져 있음. binary로 쓰면 포함안되어있음
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) #델타값이 안주어져있으니까 디폴트

total_batch = len(data_loader)
print('총 배치의 수 : {}'.format(total_batch))
#총 배치의 수 : 600
#배치 크기를 100으로 했으므로 결국 훈련 데이터는 총 60,000개

for epoch in range(training_epochs):
    avg_cost = 0

    for X, Y in data_loader: # 미니 배치 단위로 꺼내온다. X는 미니 배치, Y는 레이블.
        # image is already size of (28x28), no reshape
        # label is not one-hot encoded
        X = X.to(device)
        Y = Y.to(device)

        optimizer.zero_grad()
        hypothesis = model(X)
        cost = criterion(hypothesis, Y)
        cost.backward()
        optimizer.step()

        avg_cost += cost / total_batch

    print('[Epoch: {:>4}] cost = {:>.9}'.format(epoch + 1, avg_cost))

# 학습을 진행하지 않을 것이므로 torch.no_grad()
with torch.no_grad():
    X_test = mnist_test.test_data.view(len(mnist_test), 1, 28, 28).float().to(device)
    Y_test = mnist_test.test_labels.to(device)

    prediction = model(X_test)
    correct_prediction = torch.argmax(prediction, 1) == Y_test
    accuracy = correct_prediction.float().mean()
    print('Accuracy:', accuracy.item())

    # MNIST 테스트 데이터에서 무직위 하나 뽑아 예측
    r = random.randint(0, len(mnist_test) - 1)
    X_single_data = (
        mnist_test.test_data[r : r + 1].view(1, 1, 28, 28).float().to(device)
    )
    Y_single_data = mnist_test.test_labels[r : r + 1].to(device)

    print("Label: ", Y_single_data.item())
    single_prediction = model(X_single_data)
    print("Prediction: ", torch.argmax(single_prediction, 1).item())

    plt.imshow(
        mnist_test.test_data[r : r + 1].view(28, 28),
        cmap="Greys",
        interpolation="nearest",
    )
    plt.show()