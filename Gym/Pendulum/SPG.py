import gym
import numpy
import torch
import random
from torch.utils.tensorboard import SummaryWriter

class SPGNetWork(torch.nn.Module):
    def __init__(self, device) -> None:
        super(SPGNetWork,self).__init__()
        self.device = device
        self.initialize()

        ActorNetWorkMu = torch.nn.Sequential(
            torch.nn.Linear(3, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1),
            torch.nn.Tanh()
        )
        ActorNetWorkSigma = torch.nn.Sequential(
            torch.nn.Linear(3, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1),
            torch.nn.Softplus()
        )
        self.ActorNetWorkMu = ActorNetWorkMu.to(self.device)
        self.ActorNetWorkSigma = ActorNetWorkSigma.to(self.device)

        TargetActorNetWorkMu = torch.nn.Sequential(
            torch.nn.Linear(3, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1),
            torch.nn.Tanh()
        )
        TargetActorNetWorkSigma = torch.nn.Sequential(
            torch.nn.Linear(3, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1),
            torch.nn.Softplus()
        )
        self.TargetActorNetWorkMu = TargetActorNetWorkMu.to(self.device)
        self.TargetActorNetWorkSigma = TargetActorNetWorkSigma.to(self.device)

        CriticNetWork = torch.nn.Sequential(
            torch.nn.Linear(4, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1)
        )
        self.CriticNetWork = CriticNetWork.to(self.device)

        TargetCriticNetWork = torch.nn.Sequential(
            torch.nn.Linear(4, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1)
        )
        self.TargetCriticNetWork = TargetCriticNetWork.to(self.device)

        self.lossFunction = torch.nn.MSELoss()
        self.criticOptimizer = torch.optim.Adam(self.CriticNetWork.parameters(), lr=0.001)
        self.actorOptimizer = torch.optim.Adam(list(self.ActorNetWorkMu.parameters()) + list(self.ActorNetWorkSigma.parameters()), lr=0.0001)
        self.InitialParameter()

    def initialize(self):
        self.reward = []
        self.actorLoss = []
        self.criticLoss = []

    def InitialParameter(self):
        self.TargetActorNetWorkMu.load_state_dict(self.ActorNetWorkMu.state_dict())
        self.TargetActorNetWorkSigma.load_state_dict(self.ActorNetWorkSigma.state_dict())
        self.TargetCriticNetWork.load_state_dict(self.CriticNetWork.state_dict())

    def actorForward(self, state):
        Mu, Sigma = self.ActorNetWorkMu(state) * 2, self.ActorNetWorkSigma(state) + 1e-5
        distribution = torch.distributions.Normal(Mu, Sigma)
        action = distribution.sample().clamp(-2, 2)
        return action, distribution.log_prob(action)
    
    def criticForward(self, state, action):
        value = self.CriticNetWork(torch.cat((state, action), 1))
        return value

    def TargetCriticForward(self, state, action):
        value = self.TargetCriticNetWork(torch.cat((state, action), 1))
        return value
    
    def TargetActorForward(self, state):
        Mu, Sigma = self.TargetActorNetWorkMu(state) * 2, self.TargetActorNetWorkSigma(state) + 1e-5
        distribution = torch.distributions.Normal(Mu, Sigma)
        action = distribution.sample().clamp(-2, 2)
        return action

    def UpdateTargetNetWork(self, TargetNetWork, NetWork):
        for targetParam, param in zip(TargetNetWork.parameters(), NetWork.parameters()):
            targetParam.data.copy_(0.005 * param.data + (1 - 0.005) * targetParam.data)

    def SPGTrain(self, dataPool):
        chosenData = random.sample(dataPool, 200)
        state = torch.tensor(numpy.array([data[0] for data in chosenData]), dtype=torch.float32).reshape(-1, 3).to(self.device)
        action = torch.tensor(numpy.array([data[1] for data in chosenData]), dtype=torch.float32).reshape(-1, 1).to(self.device)
        reward = torch.tensor(numpy.array([data[2] for data in chosenData]), dtype=torch.float32).reshape(-1, 1).to(self.device)
        state_ = torch.tensor(numpy.array([data[3] for data in chosenData]), dtype=torch.float32).reshape(-1, 3).to(self.device)
        over = torch.tensor(numpy.array([data[4] for data in chosenData]), dtype=torch.float32).reshape(-1, 1).to(self.device)

        QValue = self.criticForward(state, action)

        action_ = self.TargetActorForward(state_)
        QValue_ = self.TargetCriticForward(state_, action_.reshape(-1, 1).to(self.device)).max(dim=1)[0].reshape(-1, 1)
        Target = reward + 0.9 * QValue_ * (1 - over)
        CriticLoss = self.lossFunction(QValue, Target)

        self.criticOptimizer.zero_grad()
        CriticLoss.backward()
        self.criticOptimizer.step()
        self.criticLoss.append(CriticLoss.item())

        actionPre, logProb = self.actorForward(state)
        QValue = self.criticForward(state, actionPre.reshape(-1, 1).to(self.device))
        ActorLoss = -(logProb * QValue).mean()

        self.actorOptimizer.zero_grad()
        ActorLoss.backward()
        self.actorOptimizer.step()
        self.actorLoss.append(ActorLoss.item())

        self.UpdateTargetNetWork(self.TargetActorNetWorkMu, self.ActorNetWorkMu)
        self.UpdateTargetNetWork(self.TargetActorNetWorkSigma, self.ActorNetWorkSigma)
        self.UpdateTargetNetWork(self.TargetCriticNetWork, self.CriticNetWork)

def SPG(writer):
    SPGNet = SPGNetWork("cuda")
    environment = gym.make('Pendulum-v1', render_mode="rgb_array")
    dataPool = []

    while len(dataPool) < 1000:
        state = environment.reset()[0]
        over = False
        while not over:
            action, *_ = SPGNet.actorForward(torch.tensor(state, dtype=torch.float32).to(SPGNet.device))
            state_, reward, truncated, terminated, info = environment.step(action.detach().cpu().numpy())
            over = truncated or terminated
            dataPool.append((state, action.detach().cpu().numpy(), reward, state_, over))
            state = state_

    for epoch in range(10000):
        random.shuffle(dataPool)
        writer.add_scalar('reward-epoch', play(environment, SPGNet, dataPool, epoch), epoch)
    
    writer.close()

def play(environment, SPGNet, dataPool, epoch):
    state = environment.reset()[0]
    over = False
    while not over:
        action, *_ = SPGNet.actorForward(torch.tensor(state, dtype=torch.float32).to(SPGNet.device))
        state_, reward, truncated, terminated, info = environment.step(action.detach().cpu().numpy())
        over = truncated or terminated
        dataPool.append((state, action.detach().cpu().numpy(), reward, state_, over))

        state = state_
        SPGNet.SPGTrain(dataPool)
        SPGNet.reward.append(reward)
    
    while len(dataPool) > 100000:
        dataPool.pop(0)
    
    print(f'Epoch: {epoch}, ActorLoss: {sum(SPGNet.actorLoss)}, CriticLoss: {sum(SPGNet.criticLoss)}, sumReward: {sum(SPGNet.reward)}')
    totalReward = sum(SPGNet.reward)
    SPGNet.initialize()

    return totalReward

def main():
    writer = SummaryWriter('C:\\Users\\60520\\Desktop\\RL-learning\\Log\\Pendulum-SPG')  
    SPG(writer)

if __name__ == "__main__":
    main()