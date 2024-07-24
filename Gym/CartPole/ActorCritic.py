import gym
import numpy
import torch
from torch.utils.tensorboard import SummaryWriter

class ActorCriticNetwork(torch.nn.Module):
    def __init__(self, device) -> None:
        super(ActorCriticNetwork,self).__init__()
        self.device = device
        self.initialize()

        ActorNetWork = torch.nn.Sequential(
            torch.nn.Linear(4, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 2),
            torch.nn.Softmax(dim=0)
        )
        self.ActorNetWork = ActorNetWork.to(self.device)

        ValueNetWork = torch.nn.Sequential(
            torch.nn.Linear(4, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1)
        )
        self.ValueNetWork = ValueNetWork.to(self.device)

        self.lossFunction = torch.nn.MSELoss()
        self.actorOptimizer = torch.optim.Adam(self.ActorNetWork.parameters(), lr=0.0001)
        self.valueOptimizer = torch.optim.Adam(self.ValueNetWork.parameters(), lr=0.001)

    def initialize(self):
        self.reward = []
        self.actorLoss = []
        self.criticLoss = []
    
    def actorForward(self, state):
        actionProb = self.ActorNetWork(state)
        distribution = torch.distributions.Categorical(actionProb)
        action = distribution.sample().reshape(-1)
        logProb = distribution.log_prob(action).reshape(-1)
        return action, logProb

    def criticForward(self, state):
        value = self.ValueNetWork(state)
        return value
    
    def ActorCriticTrain(self, state, reward, state_, over, logProb):
        state = torch.tensor(state, dtype=torch.float32).reshape(-1).to(self.device)
        reward = torch.tensor(reward, dtype=torch.float32).reshape(-1).to(self.device)
        state_ = torch.tensor(state_, dtype=torch.float32).reshape(-1).to(self.device)
        over = torch.tensor(over, dtype=torch.float32).reshape(-1).to(self.device)


        Value = self.criticForward(state)
        Value_ = self.criticForward(state_)
        Target = reward + 0.99 * Value_ * (1 - over)

        CriticLoss = self.lossFunction(Value, Target)
        self.valueOptimizer.zero_grad()
        CriticLoss.backward()
        self.valueOptimizer.step()
        self.criticLoss.append(CriticLoss.item())

        ActorLoss = logProb * (Value - Target).detach() # 注意正负号区别梯度上升还是梯度下降
        self.actorOptimizer.zero_grad()
        ActorLoss.backward()
        self.actorOptimizer.step()
        self.actorLoss.append(ActorLoss.item())

def ActorCritic(writer):
    ActorCriticNet = ActorCriticNetwork("cpu")
    environment = gym.make('CartPole-v1', render_mode="rgb_array")

    for epoch in range(4000):
        writer.add_scalar('reward-epoch', play(environment, ActorCriticNet, epoch), epoch)
    
    writer.close()

def play(environment, ActorCriticNet, epoch):
    state = environment.reset()[0]
    over = False
    while not over:
        action, logProb = ActorCriticNet.actorForward(torch.tensor(state, dtype=torch.float32).to(ActorCriticNet.device))
        state_, reward, truncated, terminated, info = environment.step(action.item())
        over = truncated or terminated

        ActorCriticNet.ActorCriticTrain(state, reward, state_, over, logProb)
        state = state_
        ActorCriticNet.reward.append(reward)

    print(f'Epoch: {epoch}, ActorLoss: {sum(ActorCriticNet.actorLoss)}, CriticLoss: {sum(ActorCriticNet.criticLoss)}, sumReward: {sum(ActorCriticNet.reward)}')
    totalReward = sum(ActorCriticNet.reward)
    ActorCriticNet.initialize()

    return totalReward

def main():
    writer = SummaryWriter('C:\\Users\\60520\\Desktop\\RL-learning\\Log\\CartPole-AC')  
    ActorCritic(writer)

if __name__ == "__main__":
    main()