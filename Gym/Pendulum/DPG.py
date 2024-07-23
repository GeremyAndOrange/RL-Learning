import gym
import numpy
import torch
import random
from torch.utils.tensorboard import SummaryWriter

EPSILON = 0.999

class DPGNetWork(torch.nn.Module):
    def __init__(self, device, epsilon) -> None:
        super(DPGNetWork,self).__init__()
        self.device = device

        ActorNetWork = torch.nn.Sequential(
            torch.nn.Linear(3, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1),
            torch.nn.Tanh()
        )
        self.ActorNetWork = ActorNetWork.to(self.device)

        CriticNetWork = torch.nn.Sequential(
            torch.nn.Linear(4, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1)
        )
        self.CriticNetWork = CriticNetWork.to(self.device)

        TargetActorNetWork = torch.nn.Sequential(
            torch.nn.Linear(3, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1),
            torch.nn.Tanh()
        )
        self.TargetActorNetWork = TargetActorNetWork.to(self.device)

        TargetCriticNetWork = torch.nn.Sequential(
            torch.nn.Linear(4, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1)
        )
        self.TargetCriticNetWork = TargetCriticNetWork.to(self.device)

        self.lossFunction = torch.nn.MSELoss()
        self.actorOptimizer = torch.optim.Adam(self.ActorNetWork.parameters(), lr=0.0001)
        self.valueOptimizer = torch.optim.Adam(self.CriticNetWork.parameters(), lr=0.001)
        self.epsilon = epsilon
        self.InitialParameter()
        self.initialize()

    def initialize(self):
        self.reward = []
        self.actorLoss = []
        self.criticLoss = []
    
    def InitialParameter(self):
        self.TargetActorNetWork.load_state_dict(self.ActorNetWork.state_dict())
        self.TargetCriticNetWork.load_state_dict(self.CriticNetWork.state_dict())

    def actorForward(self, state):
        action = self.ActorNetWork(state)
        return action * 2
    
    def criticForward(self, state, action):
        value = self.CriticNetWork(torch.cat((state, action), 1))
        return value

    def TargetActorForward(self, state):
        action = self.TargetActorNetWork(state)
        return action * 2
    
    def TargetCriticForward(self, state, action):
        value = self.TargetCriticNetWork(torch.cat((state, action), 1))
        return value

    def getAction(self, state):
        if random.random() < self.epsilon:
            self.epsilon = max(self.epsilon * 0.9995, 0.01)
            action = torch.tensor(numpy.random.uniform(-2, 2, 1), dtype=torch.float32).to(self.device)
        else:
            action = self.actorForward(state)
        return action
    
    def UpdateTargetNetWork(self, TargetNetWork, NetWork):
        for targetParam, param in zip(TargetNetWork.parameters(), NetWork.parameters()):
            targetParam.data.copy_(0.005 * param.data + (1 - 0.005) * targetParam.data)

    def DPGTrain(self, dataPool):
        chosenData = random.sample(dataPool, 64)
        state = torch.tensor(numpy.array([data[0] for data in chosenData]), dtype=torch.float32).reshape(-1, 3).to(self.device)
        action = torch.tensor(numpy.array([data[1] for data in chosenData]), dtype=torch.float32).reshape(-1, 1).to(self.device)
        reward = torch.tensor(numpy.array([data[2] for data in chosenData]), dtype=torch.float32).reshape(-1, 1).to(self.device)
        state_ = torch.tensor(numpy.array([data[3] for data in chosenData]), dtype=torch.float32).reshape(-1, 3).to(self.device)
        over = torch.tensor(numpy.array([data[4] for data in chosenData]), dtype=torch.float32).reshape(-1, 1).to(self.device)

        QValue = self.criticForward(state, action)

        action_ = self.TargetActorForward(state_)
        QValue_ = self.TargetCriticForward(state_, action_.reshape(-1, 1).to(self.device)).max(dim=1)[0].reshape(-1, 1)
        Target = reward + 0.99 * QValue_ * (1 - over)
        CriticLoss = self.lossFunction(QValue, Target)

        self.valueOptimizer.zero_grad()
        CriticLoss.backward()
        self.valueOptimizer.step()
        self.criticLoss.append(CriticLoss.item())

        actionPre = self.actorForward(state)
        QValue = self.criticForward(state, actionPre.reshape(-1, 1).to(self.device))
        ActorLoss = -(QValue).mean()

        self.actorOptimizer.zero_grad()
        ActorLoss.backward()
        self.actorOptimizer.step()
        self.actorLoss.append(ActorLoss.item())

        self.UpdateTargetNetWork(self.TargetActorNetWork, self.ActorNetWork)
        self.UpdateTargetNetWork(self.TargetCriticNetWork, self.CriticNetWork)

def DPG(writer):
    DPGNet = DPGNetWork("cuda", EPSILON)
    environment = gym.make('Pendulum-v1', render_mode="rgb_array")
    dataPool = []

    while len(dataPool) < 200:
        state = environment.reset()[0]
        over = False
        while not over:
            action = DPGNet.getAction(torch.tensor(state, dtype=torch.float32).to(DPGNet.device)).detach().cpu().numpy()
            state_, reward, truncated, terminated, info = environment.step(action)
            over = truncated or terminated
            dataPool.append((state, action, reward, state_, over))
            state = state_
    
    for epoch in range(1000):
        random.shuffle(dataPool)
        writer.add_scalar('reward-epoch', play(environment, DPGNet, dataPool, epoch), epoch)
    
    writer.close()

def play(environment, DPGNet, dataPool, epoch):
    state = environment.reset()[0]
    over = False
    while not over:
        action = DPGNet.getAction(torch.tensor(state, dtype=torch.float32).to(DPGNet.device)).detach().cpu().numpy()
        state_, reward, truncated, terminated, info = environment.step(action)
        over = truncated or terminated
        dataPool.append((state, action, reward, state_, over))

        state = state_
        DPGNet.DPGTrain(dataPool)
        DPGNet.reward.append(reward)
    
    while len(dataPool) > 10000:
        dataPool.pop(0)
    
    print(f'Epoch: {epoch}, ActorLoss: {sum(DPGNet.actorLoss)}, CriticLoss: {sum(DPGNet.criticLoss)}, sumReward: {sum(DPGNet.reward)}')
    totalReward = sum(DPGNet.reward)
    DPGNet.initialize()

    return totalReward

def main():
    writer = SummaryWriter('C:\\Users\\60520\\Desktop\\RL-learning\\Log\\Pendulum-DPG')  
    DPG(writer)

if __name__ == "__main__":
    main()