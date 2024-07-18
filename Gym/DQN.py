import gym
import numpy
import torch
import random

class DeePQNetwork(torch.nn.Module):
    def __init__(self, device) -> None:
        super(DeePQNetwork,self).__init__()
        self.device = device
        self.initialize()

        netWork = torch.nn.Sequential(
            torch.nn.Linear(4, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 2)
        )
        self.netWork = netWork.to(self.device)

        self.lossFunction = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.netWork.parameters(), lr=0.002)

    def initialize(self):
        self.Reward = []
        self.Loss = []
    
    def forward(self, state):
        return self.netWork(state)
    
    def saveModel(self, path):
        torch.save(self.state_dict(), path)

    def loadModel(self, path):
        self.load_state_dict(torch.load(path))
    
    def GetAction(self, state, epsilon):
        if numpy.random.random() < epsilon:
            chooseAction = numpy.random.randint(0, 2)
        else:
            state = torch.tensor(state, dtype=torch.float32).to(self.device)
            action = self.forward(state)
            chooseAction = torch.argmax(action).item()
        return chooseAction
    
    def DQNtrain(self, dataPool):
        chooseData = random.sample(dataPool, 100)
        state = torch.tensor([data[0] for data in chooseData], dtype=torch.float32).to(self.device)
        action = torch.tensor([data[1] for data in chooseData], dtype=torch.int64).to(self.device)
        reward = torch.tensor([data[2] for data in chooseData], dtype=torch.float32).to(self.device)
        state_ = torch.tensor([data[3] for data in chooseData], dtype=torch.float32).to(self.device)
        over = torch.tensor([data[4] for data in chooseData], dtype=torch.float32).to(self.device)

        QValue = self.forward(state).gather(1, action.unsqueeze(-1)).reshape(1,-1).squeeze(0)
        with torch.no_grad():
            QValue_ = self.forward(state_).max(dim=1)[0]
            Target = reward + 0.9 * QValue_ * (1 - over)
        loss = self.lossFunction(QValue, Target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.Loss.append(loss.item())

def DQN():
    DeePQNet = DeePQNetwork("cuda")
    environment = gym.make('CartPole-v1', render_mode="rgb_array")
    dataPool = []
    
    while len(dataPool) < 200:
        state = environment.reset()[0]
        over = False
        while not over:
            action = DeePQNet.GetAction(state, 0.1)
            state_, reward, truncated, terminated, info = environment.step(action)
            over = terminated or truncated
            dataPool.append((state, action, reward, state_, over))
            state = state_

    for epoch in range(10000):
        DeePQNet.DQNtrain(dataPool)
        if epoch % 100 == 0:
            testEnvironment = gym.make('CartPole-v1', render_mode="rgb_array")
        else:
            testEnvironment = gym.make('CartPole-v1', render_mode="rgb_array")
        play(testEnvironment, DeePQNet, dataPool, epoch)
            

def play(environment, DeePQNet, dataPool, epoch):
    state = environment.reset()[0]
    over, stepNumber = False, 0
    while not over:
        action = DeePQNet.GetAction(state, 0)
        state_, reward, truncated, terminated, info = environment.step(action)
        over = terminated or truncated
        dataPool.append((state, action, reward, state_, over))

        state = state_
        DeePQNet.Reward.append(reward)
        stepNumber += 1

    while len(dataPool) > 10000:
        dataPool.pop(0)

    print(f'Epoch: {epoch}, Loss: {sum(DeePQNet.Loss)}, sumReward: {sum(DeePQNet.Reward)}')
    DeePQNet.initialize()

DQN()