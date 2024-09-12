import gym
import numpy
import torch
import random
from torch.utils.tensorboard import SummaryWriter

EPSILON = 0.999

class DeePQNetwork(torch.nn.Module):
    def __init__(self, device, epsilon) -> None:
        super(DeePQNetwork,self).__init__()
        self.device = device
        self.initialize()

        advantageNetWork = torch.nn.Sequential(
            torch.nn.Linear(4, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 2)
        )
        self.advantageNetWork = advantageNetWork.to(self.device)

        valueNetWork = torch.nn.Sequential(
            torch.nn.Linear(4, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1)
        )
        self.valueNetWork = valueNetWork.to(self.device)

        self.lossFunction = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(list(self.advantageNetWork.parameters()) + list(self.valueNetWork.parameters()), lr=0.001)
        self.epsilon = epsilon

    def initialize(self):
        self.Reward = []
        self.Loss = []
    
    def forward(self, state):
        advantage = self.advantageNetWork(state)
        value = self.valueNetWork(state)
        QValue = value + advantage - advantage.mean()
        return QValue
    
    def saveModel(self, path):
        torch.save(self.state_dict(), path)

    def loadModel(self, path):
        self.load_state_dict(torch.load(path))
    
    def GetAction(self, state):
        if random.random() < self.epsilon:
            chosenAction = random.randint(0, 1)
        else:
            state = torch.tensor(state, dtype=torch.float32).to(self.device)
            action = self.forward(state)
            chosenAction = torch.argmax(action).item()
        return chosenAction
    
    def DQNtrain(self, dataPool):
        chosenData = random.sample(dataPool, 200)
        state = torch.tensor(numpy.array([data[0] for data in chosenData]), dtype=torch.float32).to(self.device)
        action = torch.tensor(numpy.array([data[1] for data in chosenData]), dtype=torch.int64).to(self.device)
        reward = torch.tensor(numpy.array([data[2] for data in chosenData]), dtype=torch.float32).to(self.device)
        state_ = torch.tensor(numpy.array([data[3] for data in chosenData]), dtype=torch.float32).to(self.device)
        over = torch.tensor(numpy.array([data[4] for data in chosenData]), dtype=torch.float32).to(self.device)

        QValue = self.forward(state).gather(1, action.unsqueeze(-1)).reshape(1,-1).squeeze(0)

        QValue_ = self.forward(state_).max(dim=1)[0]
        Target = reward + 0.99 * QValue_ * (1 - over)
        loss = self.lossFunction(QValue, Target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.Loss.append(loss.item())

def DQN(writer):
    DeePQNet = DeePQNetwork("cuda", EPSILON)
    environment = gym.make('CartPole-v1', render_mode="rgb_array")
    dataPool = []
    
    while len(dataPool) < 200:
        state = environment.reset()[0]
        over = False
        while not over:
            action = DeePQNet.GetAction(state)
            state_, reward, truncated, terminated, info = environment.step(action)
            over = truncated or terminated
            dataPool.append((state, action, reward, state_, over))
            state = state_

    for epoch in range(8000):
        DeePQNet.epsilon = max(DeePQNet.epsilon * 0.997, 0.01)
        DeePQNet.DQNtrain(dataPool)
        writer.add_scalar('reward-epoch', play(environment, DeePQNet, dataPool, epoch), epoch)
    
    writer.close()

def play(environment, DeePQNet, dataPool, epoch):
    state = environment.reset()[0]
    over, stepNumber = False, 0
    while not over:
        action = DeePQNet.GetAction(state)
        state_, reward, truncated, terminated, info = environment.step(action)
        over = truncated or (sum(DeePQNet.Reward) >= 999)
        dataPool.append((state, action, reward, state_, over))

        state = state_
        DeePQNet.Reward.append(reward)
        stepNumber += 1

    while len(dataPool) > 10000:
        dataPool.pop(0)

    print(f'Epoch: {epoch}, Loss: {sum(DeePQNet.Loss)}, sumReward: {sum(DeePQNet.Reward)}')
    sumReaward = sum(DeePQNet.Reward)
    DeePQNet.initialize()

    return sumReaward

def main():
    writer = SummaryWriter('C:\\Users\\60520\\Desktop\\RL-learning\\Gym\\Log\\CartPole-DQN')  
    DQN(writer)

if __name__ == "__main__":
    main()