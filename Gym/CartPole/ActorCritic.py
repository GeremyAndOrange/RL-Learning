import gym
import numpy
import torch
import random
import matplotlib.pyplot

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

        CriticNetWork = torch.nn.Sequential(
            torch.nn.Linear(5, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1)
        )
        self.CriticNetWork = CriticNetWork.to(self.device)

        self.lossFunction = torch.nn.MSELoss()
        self.actorOptimizer = torch.optim.Adam(self.ActorNetWork.parameters(), lr=0.003)
        self.valueOptimizer = torch.optim.Adam(self.CriticNetWork.parameters(), lr=0.003)

    def initialize(self):
        self.reward = []
        self.actorLoss = []
        self.criticLoss = []
    
    def actorForward(self, state):
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        actionProb = self.ActorNetWork(state)
        distribution = torch.distributions.Categorical(actionProb)
        action = distribution.sample()
        logProb = distribution.log_prob(action).reshape(-1, 1)
        return numpy.array(action), logProb

    def criticForward(self, state, action):
        feature = torch.cat((state, action), dim=1)
        value = self.CriticNetWork(feature)
        return value
    
    def saveModel(self, path):
        torch.save(self.ActorNetWork.state_dict(), path)
        torch.save(self.CriticNetWork.state_dict(), path)

    def loadModel(self, path):
        self.ActorNetWork.load_state_dict(torch.load(path))
        self.CriticNetWork.load_state_dict(torch.load(path))
    
    def ActorCriticTrain(self, dataPool):
        chosenData = random.sample(dataPool, 200)
        state = torch.tensor(numpy.array([data[0] for data in chosenData]), dtype=torch.float32).reshape(-1, 4).to(self.device)
        action = torch.tensor(numpy.array([data[1] for data in chosenData]), dtype=torch.int64).reshape(-1, 1).to(self.device)
        reward = torch.tensor(numpy.array([data[2] for data in chosenData]), dtype=torch.float32).reshape(-1, 1).to(self.device)
        state_ = torch.tensor(numpy.array([data[3] for data in chosenData]), dtype=torch.float32).reshape(-1, 4).to(self.device)
        over = torch.tensor(numpy.array([data[4] for data in chosenData]), dtype=torch.float32).reshape(-1, 1).to(self.device)

        QValue = self.criticForward(state, action)

        action_, _ = self.actorForward(state_)
        QValue_ = self.criticForward(state_, torch.tensor(action_, dtype=torch.int64).reshape(-1, 1).to(self.device)).max(dim=1)[0].reshape(-1, 1)
        Target = reward + 0.99 * QValue_ * (1 - over)
        CriticLoss = self.lossFunction(QValue, Target)

        self.valueOptimizer.zero_grad()
        CriticLoss.backward()
        self.valueOptimizer.step()
        self.criticLoss.append(CriticLoss.item())

        actionPre, logProb = self.actorForward(state)
        QValue = self.criticForward(state, torch.tensor(actionPre, dtype=torch.int64).reshape(-1, 1).to(self.device))
        ActorLoss = -(logProb * QValue).mean()

        self.actorOptimizer.zero_grad()
        ActorLoss.backward()
        self.actorOptimizer.step()
        self.actorLoss.append(ActorLoss.item())

def ActorCritic():
    ActorCriticNet = ActorCriticNetwork("cpu")
    environment = gym.make('CartPole-v1', render_mode="rgb_array")
    dataPool = []

    while len(dataPool) < 200:
        state = environment.reset()[0]
        over = False
        while not over:
            action, _ = ActorCriticNet.actorForward(state)
            state_, reward, truncated, terminated, info = environment.step(action)
            over = truncated or terminated
            dataPool.append((state, action, reward, state_, over))
            state = state_

    record = []
    for epoch in range(10000):
        random.shuffle(dataPool)
        ActorCriticNet.ActorCriticTrain(dataPool)
        record.append((epoch, play(environment, ActorCriticNet, dataPool, epoch)))
    
    Draw(record)

def play(environment, ActorCriticNet, dataPool, epoch):
    state = environment.reset()[0]
    over = False
    while not over:
        action, _ = ActorCriticNet.actorForward(state)
        state_, reward, truncated, terminated, info = environment.step(action)
        over = truncated or terminated
        dataPool.append((state, action, reward, state_, over))

        state = state_
        ActorCriticNet.reward.append(reward)
    
    while len(dataPool) > 10000:
        dataPool.pop(0)
    
    print(f'Epoch: {epoch}, ActorLoss: {sum(ActorCriticNet.actorLoss)}, CriticLoss: {sum(ActorCriticNet.criticLoss)}, sumReward: {sum(ActorCriticNet.reward)}')
    totalReward = sum(ActorCriticNet.reward)
    ActorCriticNet.initialize()

    return totalReward

def Draw(record):
    x = [coord[0] for coord in record]
    y = [coord[1] for coord in record]
    matplotlib.pyplot.plot(x, y, '.')
    matplotlib.pyplot.xlabel('Epoch')
    matplotlib.pyplot.ylabel('Sum of Rewards')
    matplotlib.pyplot.title('Training Performance')
    matplotlib.pyplot.show()

    print(f'Average Reward: {sum(y)/len(y)}, Max Reward: {max(y)}, Min Reward: {min(y)}')

ActorCritic()