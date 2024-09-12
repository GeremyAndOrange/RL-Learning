import gym
import torch
import numpy
import random
from torch.utils.tensorboard import SummaryWriter

EPSILON = 0.999

class DoubleDQNetwork(torch.nn.Module):
    def __init__(self, device, epsilon) -> None:
        super(DoubleDQNetwork,self).__init__()
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

        targetAdvantageNetWork = torch.nn.Sequential(
            torch.nn.Linear(4, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 2)
        )
        self.targetAdvantageNetWork = targetAdvantageNetWork.to(self.device)

        targetValueNetWork = torch.nn.Sequential(
            torch.nn.Linear(4, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1)
        )
        self.targetValueNetWork = targetValueNetWork.to(self.device)

        self.lossFunction = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(list(self.advantageNetWork.parameters()) + list(self.valueNetWork.parameters()), lr=0.001)
        self.epsilon = epsilon

    def initialize(self):
        self.Reward = []
        self.Loss = []
    
    def UpdateTargetModel(self):
        self.targetAdvantageNetWork.load_state_dict(self.advantageNetWork.state_dict())
        self.targetValueNetWork.load_state_dict(self.valueNetWork.state_dict())

    def DQNforward(self, state):
        advantage = self.advantageNetWork(state)
        value = self.valueNetWork(state)
        QValue = value + advantage - advantage.mean()
        return QValue
    
    def targetDQNforward(self, state):
        advantage = self.targetAdvantageNetWork(state)
        value = self.targetValueNetWork(state)
        TargetQValue = value + advantage - advantage.mean()
        return TargetQValue
    
    def saveModel(self, path):
        torch.save(self.state_dict(), path)

    def loadModel(self, path):
        self.load_state_dict(torch.load(path))
    
    def GetAction(self, state):
        if random.random() < self.epsilon:
            chooseAction = random.randint(0, 1)
        else:
            state = torch.tensor(state, dtype=torch.float32).to(self.device)
            action = self.DQNforward(state)
            chooseAction = torch.argmax(action).item()
        return chooseAction
    
    def DQNtrain(self, dataPool, epoch):
        chosenData = random.sample(dataPool, 200)
        state = torch.tensor(numpy.array([data[0] for data in chosenData]), dtype=torch.float32).to(self.device)
        action = torch.tensor(numpy.array([data[1] for data in chosenData]), dtype=torch.int64).to(self.device)
        reward = torch.tensor(numpy.array([data[2] for data in chosenData]), dtype=torch.float32).to(self.device)
        state_ = torch.tensor(numpy.array([data[3] for data in chosenData]), dtype=torch.float32).to(self.device)
        over = torch.tensor(numpy.array([data[4] for data in chosenData]), dtype=torch.float32).to(self.device)

        QValue = self.DQNforward(state).gather(1, action.unsqueeze(-1)).reshape(1,-1).squeeze(0)
        with torch.no_grad():
            action_ = self.DQNforward(state_).argmax(dim=1)
            QValue_ = self.targetDQNforward(state_).gather(1, action_.unsqueeze(-1)).reshape(1,-1).squeeze(0)
            Target = reward + 0.99 * QValue_ * (1 - over)
        loss = self.lossFunction(QValue, Target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.Loss.append(loss.item())

        if epoch % 6 == 0:
            self.UpdateTargetModel()

def DoubleDQN(writer):
    DoubleDQNet = DoubleDQNetwork("cuda", EPSILON)
    environment = gym.make('CartPole-v1', render_mode="rgb_array")
    dataPool = []
    
    while len(dataPool) < 200:
        state = environment.reset()[0]
        over = False
        while not over:
            action = DoubleDQNet.GetAction(state)
            state_, reward, truncated, terminated, info = environment.step(action)
            over = truncated or terminated
            dataPool.append((state, action, reward, state_, over))
            state = state_

    for epoch in range(8000):
        DoubleDQNet.epsilon = max(DoubleDQNet.epsilon * 0.9975, 0.1)
        DoubleDQNet.DQNtrain(dataPool, epoch)
        writer.add_scalar('reward-epoch', play(environment, DoubleDQNet, dataPool, epoch), epoch)
    
    writer.close()

def play(environment, DoubleDQNet, dataPool, epoch):
    state = environment.reset()[0]
    over, stepNumber = False, 0
    while not over:
        action = DoubleDQNet.GetAction(state)
        state_, reward, truncated, terminated, info = environment.step(action)
        over = truncated or (sum(DoubleDQNet.Reward) >= 999)
        reward = reward if not truncated else -10
        dataPool.append((state, action, reward, state_, over))

        state = state_
        DoubleDQNet.Reward.append(reward)
        stepNumber += 1

    while len(dataPool) > 10000:
        dataPool.pop(0)

    print(f'Epoch: {epoch}, Loss: {sum(DoubleDQNet.Loss)}, sumReward: {sum(DoubleDQNet.Reward)}')
    sumReaward = sum(DoubleDQNet.Reward)
    DoubleDQNet.initialize()

    return sumReaward

def main():
    writer = SummaryWriter('C:\\Users\\60520\\Desktop\\RL-learning\\Gym\\Log\\CartPole-DDQN')  
    DoubleDQN(writer)

if __name__ == "__main__":
    main()