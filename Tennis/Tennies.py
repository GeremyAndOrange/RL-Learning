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
        self.dataStoreLen = 10000

class StateFeatureNetworkl(torch.nn.Module):
    def __init__(self, device):
        super(StateFeatureNetworkl, self).__init__()
        self.device = device
        self.netWork = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, 5, 1, 1),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32, 32, 5, 1, 1),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32, 16, 5, 1, 1),
            torch.nn.MaxPool2d(2),
            torch.nn.Flatten(),
            torch.nn.Linear(6912, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 16)
        )
        self.netWork.to(self.device)
    
    def forward(self, state):
        stateFeature = self.netWork(state)
        return stateFeature

class ValueNetwork(torch.nn.Module):
    def __init__(self, device):
        super(ValueNetwork, self).__init__()
        self.device = device
        self.netWork = torch.nn.Sequential(
            torch.nn.Linear(16, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 1)
        )
        self.netWork.to(self.device)
    
    def forward(self, stateFeature):
        value = self.netWork(stateFeature)
        return value

class AdvantageNetwork(torch.nn.Module):
    def __init__(self, device):
        super(AdvantageNetwork, self).__init__()
        self.device = device
        self.netWork = torch.nn.Sequential(
            torch.nn.Linear(16, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 18)
        )
        self.netWork.to(self.device)
    
    def forward(self, stateFeature):
        advantage = self.netWork(stateFeature)
        return advantage

class DQNNetWork():
    def __init__(self, device):
        self.stateFeatureModel = StateFeatureNetworkl(device)
        self.valueModel = ValueNetwork(device)
        self.advantageModel = AdvantageNetwork(device)
        self.targetValueModel = ValueNetwork(device)
        self.targetAdvantageModel = AdvantageNetwork(device)

        self.hyperParameters = HyperParameters()
        self.optimizer = torch.optim.Adam(list(self.stateFeatureModel.parameters()) + list(self.valueModel.parameters()) + list(self.advantageModel.parameters()), lr=self.hyperParameters.lr)
        self.lossFunction = torch.nn.MSELoss()
        self.dataStore = []
        self.device = device
        self.initialize()
        self.InitialParameter()

    def initialize(self):
        self.Reward = []
        self.Loss = []

    def InitialParameter(self):
        self.targetAdvantageModel.load_state_dict(self.advantageModel.state_dict())
        self.targetValueModel.load_state_dict(self.valueModel.state_dict())

    def saveModel(self, path):
        torch.save({
            'valueModel_state_dict': self.valueModel.state_dict(),
            'advantageModel_state_dict': self.advantageModel.state_dict(),
            'targetValueModel_state_dict': self.targetValueModel.state_dict(),
            'targetAdvantageModel_state_dict': self.targetAdvantageModel.state_dict(),
        }, path)

    def loadModel(self, path):
        checkpoint = torch.load(path)
        self.valueModel.load_state_dict(checkpoint['valueModel_state_dict'])
        self.advantageModel.load_state_dict(checkpoint['advantageModel_state_dict'])
        self.targetValueModel.load_state_dict(checkpoint['targetValueModel_state_dict'])
        self.targetAdvantageModel.load_state_dict(checkpoint['targetAdvantageModel_state_dict'])

        self.valueModel.eval()
        self.advantageModel.eval()
        self.targetValueModel.eval()
        self.targetAdvantageModel.eval()

    def UpdateTargetNetWork(self, TargetNetWork, NetWork):
        for targetParam, param in zip(TargetNetWork.parameters(), NetWork.parameters()):
            targetParam.data.copy_(0.005 * param.data + (1 - 0.005) * targetParam.data)

    def forward(self, state):
        stateFeature = self.stateFeatureModel(state)
        advantage = self.advantageModel(stateFeature)
        value = self.valueModel(stateFeature)
        QValue = value + advantage - advantage.mean()
        return QValue
    
    def targetForward(self, state):
        stateFeature = self.stateFeatureModel(state)
        advantage = self.targetAdvantageModel(stateFeature)
        value = self.targetValueModel(stateFeature)
        TargetQValue = value + advantage - advantage.mean()
        return TargetQValue
    
    def getAction(self, state):
        if numpy.random.random() < self.hyperParameters.epsilon:
            chosenAction = numpy.random.randint(0, 18)
        else:
            state = torch.tensor(state, dtype=torch.float32).to(self.device).reshape(-1, 3, 210, 160)
            action = self.forward(state)
            chosenAction = torch.argmax(action).item()
        return chosenAction
    
    def DQNtrain(self):
        chosenData = random.sample(self.dataStore, 200)
        state = torch.tensor(numpy.array([data[0] for data in chosenData]), dtype=torch.float32).reshape(-1, 3, 210, 160).to(self.device)
        action = torch.tensor(numpy.array([data[1] for data in chosenData]), dtype=torch.int64).reshape(-1, 1).to(self.device)
        reward = torch.tensor(numpy.array([data[2] for data in chosenData]), dtype=torch.float32).reshape(-1, 1).to(self.device)
        state_ = torch.tensor(numpy.array([data[3] for data in chosenData]), dtype=torch.float32).reshape(-1, 3, 210, 160).to(self.device)
        over = torch.tensor(numpy.array([data[4] for data in chosenData]), dtype=torch.float32).reshape(-1, 1).to(self.device)

        QValue = self.forward(state).gather(1, action)
        with torch.no_grad():
            action_ = self.targetForward(state_).argmax(dim=1).reshape(-1, 1)
            QValue_ = self.targetForward(state_).gather(1, action_)
            Target = reward + 0.99 * QValue_ * (1 - over)
        loss = self.lossFunction(QValue, Target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.Loss.append(loss.item())

        self.UpdateTargetNetWork(self.targetValueModel, self.valueModel)
        self.UpdateTargetNetWork(self.targetAdvantageModel, self.advantageModel)

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

    def play(self, environment, epoch):
        state = environment.reset()[0]
        over = False
        while not over:
            action = self.getAction(state)
            state_, reward, truncated, terminated, info = environment.step(action)
            over = truncated or terminated
            self.dataStore.append((state, action, reward, state_, over))
            state = state_
            self.Reward.append(reward)

        sumReaward = sum(self.Reward)
        while len(self.dataStore) > self.hyperParameters.dataStoreLen:
            self.dataStore.pop(0)
        
        print(f'Epoch: {epoch}, Loss: {sum(self.Loss)}, sumReward: {sum(self.Reward)}')
        self.initialize()
        return sumReaward

def main():
    DeepQNet = DQNNetWork("cuda")
    environment = gym.make('Tennis-v4', render_mode="rgb_array")
    writer = SummaryWriter('C:\\Users\\60520\\Desktop\\RL-learning\\Log\\Tnnies-DQN')
    DeepQNet.getData(environment)

    for epoch in range(100000):
        DeepQNet.hyperParameters.epsilon = max(DeepQNet.hyperParameters.epsilon * 0.997, 0.01)
        DeepQNet.DQNtrain()
        writer.add_scalar('reward-epoch', DeepQNet.play(environment, epoch), epoch)
        if epoch + 1 % 1000 == 0:
            DeepQNet.saveModel('.\\model\\Tennis\\' + str(epoch + 1) + '.pth')

if __name__ == '__main__':
    main()