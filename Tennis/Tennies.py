import gym
import numpy
import torch
import random
from torch.utils.tensorboard import SummaryWriter

# Discrete action space, so choose DQN
class HyperParameters:
    def __init__(self):
        self.lr = 1e-3
        self.gamma = 0.99
        self.epsilon = 0.999
        self.dataStoreLen = 100000

class StateFeatureNetworkl(torch.nn.Module):
    def __init__(self, device):
        super(StateFeatureNetworkl, self).__init__()
        self.device = device
        self.netWork = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, 5, 1, 1),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32, 32, 5, 1, 1),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32, 64, 5, 1, 1),
            torch.nn.MaxPool2d(2),
            torch.nn.Flatten()
        )
        self.netWork.to(self.device)

class ValueNetwork(torch.nn.Module):
    def __init__(self, device):
        super(ValueNetwork, self).__init__()
        self.device = device
        self.netWork = torch.nn.Sequential(
            torch.nn.Linear(1, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 1)
        )
        self.netWork.to(self.device)

class AdvantageNetwork(torch.nn.Module):
    def __init__(self, device):
        super(AdvantageNetwork, self).__init__()
        self.device = device
        self.netWork = torch.nn.Sequential(
            torch.nn.Linear(1, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 18)
        )
        self.netWork.to(self.device)

class DQNNetWork():
    def __init__(self, device):
        self.stateFeatureModel = StateFeatureNetworkl(device)
        self.valueModel = ValueNetwork(device)
        self.advantageModel = AdvantageNetwork(device)
        self.hyperParameters = HyperParameters()

        self.optimizer = torch.optim.Adam(list(self.stateFeatureModel.parameters()) + list(self.valueModel.parameters()) + list(self.advantageModel.parameters()), lr=self.hyperParameters.lr)
        self.dataStore = []

    def initialize(self):
        self.Reward = []
        self.Loss = []

    def forward(self, state):
        stateFeature = self.stateFeatureModel(state)
        advantage = self.advantageModel(stateFeature)
        value = self.valueModel(stateFeature)
        QValue = value + advantage - advantage.mean()
        return QValue
    
    def getAction(self, state):
        if numpy.random.random() < self.hyperParameters.epsilon:
            chosenAction = numpy.random.randint(0, 18)
        else:
            state = torch.tensor(state, dtype=torch.float32).to(self.device)
            action = self.forward(state)
            chosenAction = torch.argmax(action).item()
        return chosenAction
    
    def DQNtrain(self):
        chosenData = random.sample(self.dataStore, 200)
        state = torch.tensor([data[0] for data in chosenData], dtype=torch.float32).to(self.device)
        action = torch.tensor([data[1] for data in chosenData], dtype=torch.int64).to(self.device)
        reward = torch.tensor([data[2] for data in chosenData], dtype=torch.float32).to(self.device)
        state_ = torch.tensor([data[3] for data in chosenData], dtype=torch.float32).to(self.device)
        over = torch.tensor([data[4] for data in chosenData], dtype=torch.float32).to(self.device)

        QValue = self.forward(state).gather(1, action.unsqueeze(-1)).reshape(1,-1).squeeze(0)
        QValue_ = self.forward(state_).max(dim=1)[0]
        Target = reward + 0.99 * QValue_ * (1 - over)
        loss = self.lossFunction(QValue, Target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.Loss.append(loss.item())

    def getData(self, environment):
        while len(self.dataStore) < self.hyperParameters.dataStoreLen:
            state = environment.reset()[0]
            over = False
            while not over:
                action = self.getAction(state)
                state_, reward, truncated, terminated, info = environment.step(action)
                over = truncated or terminated
                self.dataStore.append((state, action, reward, state_, over))
                state = state_

    def play(self, environment):
        state = environment.reset()[0]
        over = False
        while not over:
            action = self.getAction(state)
            state_, reward, truncated, terminated, info = environment.step(action)
            over = truncated or terminated
            state = state_
        return reward

def main():
    DeepQNet = DQNNetWork("cuda")
    environment = gym.make('ALE/Tennis-v5', render_mode="rgb_array")
    writer = SummaryWriter('C:\\Users\\60520\\Desktop\\RL-learning\\Log\\Tnnies-DQN') 
    DeepQNet.getData(environment)

    for epoch in range(8000):
        DeepQNet.hyperParameters.epsilon = max(DeepQNet.hyperParameters.epsilon * 0.997, 0.01)
        DeepQNet.DQNtrain()
        writer.add_scalar('reward-epoch', DeepQNet.play(environment), epoch)###

if __name__ == '__main__':
    main()