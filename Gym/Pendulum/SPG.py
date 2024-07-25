import gym
import torch
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

        ValueNetWork = torch.nn.Sequential(
            torch.nn.Linear(3, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1)
        )
        self.ValueNetWork = ValueNetWork.to(self.device)

        self.lossFunction = torch.nn.MSELoss()
        self.valuecOptimizer = torch.optim.Adam(self.ValueNetWork.parameters(), lr=0.001)
        self.actorOptimizer = torch.optim.Adam(list(self.ActorNetWorkMu.parameters()) + list(self.ActorNetWorkSigma.parameters()), lr=0.0001)

    def initialize(self):
        self.reward = []
        self.actorLoss = []
        self.criticLoss = []

    def actorForward(self, state):
        Mu, Sigma = self.ActorNetWorkMu(state) * 2, self.ActorNetWorkSigma(state) + 1e-5
        distribution = torch.distributions.Normal(Mu, Sigma)
        action = distribution.sample().clamp(-2, 2).reshape(-1)
        logProb = distribution.log_prob(action).reshape(-1)
        return action, logProb
    
    def criticForward(self, state):
        value = self.ValueNetWork(state)
        return value

    def SPGTrain(self, state, reward, state_, over, logProb):
        state = torch.tensor(state, dtype=torch.float32).reshape(-1).to(self.device)
        reward = torch.tensor(reward, dtype=torch.float32).reshape(-1).to(self.device)
        state_ = torch.tensor(state_, dtype=torch.float32).reshape(-1).to(self.device)
        over = torch.tensor(over, dtype=torch.float32).reshape(-1).to(self.device)

        Value = self.criticForward(state)
        Value_ = self.criticForward(state_)
        Target = reward + 0.99 * Value_ * (1 - over)

        CriticLoss = self.lossFunction(Value, Target)
        self.valuecOptimizer.zero_grad()
        CriticLoss.backward()
        self.valuecOptimizer.step()
        self.criticLoss.append(CriticLoss.item())

        ActorLoss = logProb * (Value - Target).detach()
        self.actorOptimizer.zero_grad()
        ActorLoss.backward()
        self.actorOptimizer.step()
        self.actorLoss.append(ActorLoss.item())

def SPG(writer):
    SPGNet = SPGNetWork("cuda")
    environment = gym.make('Pendulum-v1', render_mode="rgb_array")

    for epoch in range(4000):
        writer.add_scalar('reward-epoch', play(environment, SPGNet, epoch), epoch)
    
    writer.close()

def play(environment, SPGNet, epoch):
    state = environment.reset()[0]
    over = False
    while not over:
        action, logProb = SPGNet.actorForward(torch.tensor(state, dtype=torch.float32).to(SPGNet.device))
        state_, reward, truncated, terminated, info = environment.step(action.detach().cpu().numpy())
        over = truncated or terminated

        SPGNet.SPGTrain(state, reward, state_, over, logProb)
        state = state_
        SPGNet.reward.append(reward)
    
    print(f'Epoch: {epoch}, ActorLoss: {sum(SPGNet.actorLoss)}, CriticLoss: {sum(SPGNet.criticLoss)}, sumReward: {sum(SPGNet.reward)}')
    totalReward = sum(SPGNet.reward)
    SPGNet.initialize()

    return totalReward

def main():
    writer = SummaryWriter('C:\\Users\\60520\\Desktop\\RL-learning\\Log\\Pendulum-SPG')  
    SPG(writer)

if __name__ == "__main__":
    main()