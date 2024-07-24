import gym
import numpy
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

class PPONetWork(torch.nn.Module):
    def __init__(self, device) -> None:
        super(PPONetWork,self).__init__()
        self.device = device
        self.dataStore = []
        self.initialize()

        OldActorNetWorkMu = torch.nn.Sequential(
            torch.nn.Linear(3, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1),
            torch.nn.Tanh()
        )
        OldActorNetWorkSigma = torch.nn.Sequential(
            torch.nn.Linear(3, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1),
            torch.nn.Softplus()
        )
        NewActorNetWorkMu = torch.nn.Sequential(
            torch.nn.Linear(3, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1),
            torch.nn.Tanh()
        )
        NewActorNetWorkSigma = torch.nn.Sequential(
            torch.nn.Linear(3, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1),
            torch.nn.Softplus()
        )
        self.OldActorNetWorkMu = OldActorNetWorkMu.to(self.device)
        self.OldActorNetWorkSigma = OldActorNetWorkSigma.to(self.device)
        self.NewActorNetWorkMu = NewActorNetWorkMu.to(self.device)
        self.NewActorNetWorkSigma = NewActorNetWorkSigma.to(self.device)

        ValueNetWork = torch.nn.Sequential(
            torch.nn.Linear(3, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1)
        )
        self.ValueNetWork = ValueNetWork.to(self.device)

        self.lossFunction = torch.nn.MSELoss()
        self.valuecOptimizer = torch.optim.Adam(self.ValueNetWork.parameters(), lr=0.0003)
        self.actorOptimizer = torch.optim.Adam(list(self.NewActorNetWorkMu.parameters()) + list(self.NewActorNetWorkSigma.parameters()), lr=0.0001)
        self.InitialParameter()

    def initialize(self):
        self.actorLoss = []
        self.criticLoss = []
    
    def InitialParameter(self):
        self.NewActorNetWorkMu.load_state_dict(self.OldActorNetWorkMu.state_dict())
        self.NewActorNetWorkSigma.load_state_dict(self.OldActorNetWorkSigma.state_dict())

    def OldActorForward(self, state):
        Mu, Sigma = self.OldActorNetWorkMu(state) * 2, self.OldActorNetWorkSigma(state) + 1e-5
        distribution = torch.distributions.Normal(Mu, Sigma)
        action = distribution.sample().clamp(-2, 2).reshape(-1, 1)
        logProb = distribution.log_prob(action).reshape(-1, 1)
        return action, logProb
    
    def NewActorForward(self, state, action):
        Mu, Sigma = self.NewActorNetWorkMu(state) * 2, self.NewActorNetWorkSigma(state) + 1e-5
        distribution = torch.distributions.Normal(Mu, Sigma)
        logProb = distribution.log_prob(action).reshape(-1, 1)
        return logProb

    def ValueForward(self, state):
        value = self.ValueNetWork(state)
        return value

    def UpdateTargetNetWork(self, TargetNetWork, NetWork):
        TargetNetWork.load_state_dict(NetWork.state_dict())

    def PPOtrain(self):
        oldState = torch.tensor(numpy.array([data[0] for data in self.dataStore]), dtype=torch.float32).reshape(-1, 3).to(self.device)
        oldAction = torch.tensor(numpy.array([data[1] for data in self.dataStore]), dtype=torch.float32).reshape(-1, 1).to(self.device)
        reward = torch.tensor(numpy.array([data[2] for data in self.dataStore]), dtype=torch.float32).reshape(-1, 1).to(self.device)
        oldState_ = torch.tensor(numpy.array([data[3] for data in self.dataStore]), dtype=torch.float32).reshape(-1, 3).to(self.device)
        oldActionLogProb = torch.tensor([data[4] for data in self.dataStore], dtype=torch.float32).reshape(-1, 1).to(self.device)

        reward = (reward - reward.mean()) / (reward.std() + 1e-5)
        with torch.no_grad():
            targetValue = reward + 0.99 * self.ValueForward(oldState_)
        advantage = (targetValue - self.ValueForward(oldState)).detach()

        for _ in range(10):
            for index in BatchSampler(SubsetRandomSampler(range(len(self.dataStore))), 128, False):
                newActionLogProb = self.NewActorForward(oldState[index], oldAction[index])
                ratio = torch.exp(newActionLogProb - oldActionLogProb[index])

                L1 = ratio * advantage[index]
                L2 = torch.clamp(ratio, 1 - 0.2, 1 + 0.2) * advantage[index]
                ActorLoss = -torch.min(L1, L2).mean()
                self.actorOptimizer.zero_grad()
                ActorLoss.backward()
                torch.nn.utils.clip_grad_norm_(self.NewActorNetWorkMu.parameters(), 0.5)
                self.actorOptimizer.step()
                self.actorLoss.append(ActorLoss.item())

                ValueLoss = self.lossFunction(self.ValueForward(oldState[index]), targetValue[index])
                self.valuecOptimizer.zero_grad()
                ValueLoss.backward()
                torch.nn.utils.clip_grad_norm_(self.ValueNetWork.parameters(), 0.5)
                self.valuecOptimizer.step()
                self.criticLoss.append(ValueLoss.item())

        self.UpdateTargetNetWork(self.OldActorNetWorkMu, self.NewActorNetWorkMu)
        self.UpdateTargetNetWork(self.OldActorNetWorkSigma, self.NewActorNetWorkSigma)
        self.dataStore.clear()

def PPO(writer):
    PPONet = PPONetWork("cuda")
    environment = gym.make('Pendulum-v1', render_mode="rgb_array")
    
    for epoch in range(1000):
        writer.add_scalar('reward-epoch', play(environment, PPONet, epoch), epoch)
    
    writer.close()

def play(environment, PPONet, epoch):
    state = environment.reset()[0]
    over, totalReward = False, 0
    while not over:
        action, logProb = PPONet.OldActorForward(torch.tensor(state, dtype=torch.float32).to(PPONet.device))
        state_, reward, truncated, terminated, info = environment.step(action.reshape(-1).detach().cpu().numpy())
        over = truncated or terminated
        PPONet.dataStore.append((state, action.item(), reward, state_, logProb))
        if len(PPONet.dataStore) >= 1000:
            PPONet.PPOtrain()

        state = state_
        totalReward += reward
    
    if epoch % 10 == 0:
        print(f'Epoch: {epoch}, ActorLoss: {sum(PPONet.actorLoss)}, CriticLoss: {sum(PPONet.criticLoss)}, Reward: {totalReward}')

    PPONet.initialize()
    return totalReward

def main():
    writer = SummaryWriter('C:\\Users\\60520\\Desktop\\RL-learning\\Log\\Pendulum-PPO')  
    PPO(writer)

if __name__ == "__main__":
    main()