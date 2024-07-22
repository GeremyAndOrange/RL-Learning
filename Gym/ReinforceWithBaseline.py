import gym
import numpy
import torch
import matplotlib.pyplot

class ReinforceNetWork(torch.nn.Module):
    def __init__(self, device) -> None:
        super(ReinforceNetWork,self).__init__()
        self.device = device
        self.initialize()

        policyNetWork = torch.nn.Sequential(
            torch.nn.Linear(4, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 2),
            torch.nn.Softmax(dim=0)
        )
        self.policyNetWork = policyNetWork.to(self.device)

        valueNetWork = torch.nn.Sequential(
            torch.nn.Linear(4, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1)
        )
        self.valueNetWork = valueNetWork.to(self.device)

        self.lossFunction = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.policyNetWork.parameters(), lr=0.001)
        self.valueOptimizer = torch.optim.Adam(self.valueNetWork.parameters(), lr=0.001)
    
    def initialize(self):
        self.Reward = []
        self.Loss = []
        self.dataStore = []
    
    def forward(self, state):
        actionProbs = self.policyNetWork(state)
        distribution = torch.distributions.Categorical(actionProbs)
        action = distribution.sample()
        logProb = distribution.log_prob(action)
        return numpy.array(action), logProb
    
    def valueForward(self, state):
        return self.valueNetWork(state)

    def saveModel(self, path):
        torch.save(self.state_dict(), path)

    def loadModel(self, path):
        self.load_state_dict(torch.load(path))

    def ReinforceTrain(self):
        discountReturn, loss, valueLoss = 0, 0, 0
        futureReturn = []
        for reward, *_ in reversed(self.dataStore):
            discountReturn = reward + discountReturn * 0.99
            futureReturn.insert(0, discountReturn)

        stateList = [item[2]  for item in self.dataStore]
        value = self.valueForward(torch.tensor(numpy.array(stateList), dtype=torch.float).to(self.device)).reshape(-1)
        returns = torch.tensor(numpy.array(futureReturn), dtype=torch.float).to(self.device)
        valueLoss = self.lossFunction(value, returns)
        self.valueOptimizer.zero_grad()
        valueLoss.backward()
        self.valueOptimizer.step()

        for reward, logProb, state in reversed(self.dataStore):
            discountReturn = reward + discountReturn * 0.99
            value = self.valueForward(torch.tensor(state, dtype=torch.float).to(self.device))
            loss += -logProb * (discountReturn - value)
        
        loss = loss/len(self.dataStore)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.Loss.append(loss.item())

def Reinforce():
    ReinforceNet = ReinforceNetWork("cpu")
    environment = gym.make('CartPole-v1', render_mode="rgb_array")

    record = []
    for epoch in range(8000):
        state = environment.reset()[0]
        over = False
        while not over:
            action, logProb = ReinforceNet.forward(torch.tensor(state, dtype=torch.float).to(ReinforceNet.device))
            state_, reward, truncated, terminated, info = environment.step(action)
            over = terminated or truncated
            ReinforceNet.dataStore.append((reward, logProb, state))
            state = state_
            ReinforceNet.Reward.append(reward)
        ReinforceNet.ReinforceTrain()
        print(f'Epoch: {epoch}, Loss: {sum(ReinforceNet.Loss)}, sumReward: {sum(ReinforceNet.Reward)}')
        record.append((epoch, sum(ReinforceNet.Reward)))
        ReinforceNet.initialize()
    
    Draw(record)

def Draw(record):
    x = [coord[0] for coord in record]
    y = [coord[1] for coord in record]
    matplotlib.pyplot.plot(x, y, '.')
    matplotlib.pyplot.xlabel('Epoch')
    matplotlib.pyplot.ylabel('Sum of Rewards')
    matplotlib.pyplot.title('Training Performance')
    matplotlib.pyplot.show()

    print(f'Average Reward: {sum(y)/len(y)}, Max Reward: {max(y)}, Min Reward: {min(y)}')

Reinforce()