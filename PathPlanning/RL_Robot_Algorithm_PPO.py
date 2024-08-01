import math
import numpy
import torch
import random
import matplotlib.pyplot
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

# parameters
START = [2, 2]          # 起点坐标
END = [48, 48]          # 终点坐标
MAX_NODES = 500         # 节点最大数量
STEP_SIZE = 0.5         # 步长
OBSTACLE_NUM = 9        # 障碍物数量
OBSTACLE_RADIUS = 4     # 障碍物半径
MAP_LENGTH = 50         # 地图长度
MAP_HEIGHT = 50         # 地图宽度
MAP_RESOLUTION = 0.5    # 地图分辨率
SCAN_RANGE = 31         # 智能体扫描范围
ALPHA = 1               # 奖励权重

# enumeration value
COLLISION = 127
ROBOT = 255
OUT_MAP = -1

class HyperParameters():
    def __init__(self):
        self.actorLr = 2e-4
        self.criticLr = 3e-4
        self.gamma = 0.99
        self.gradEpsilon = 0.2
        self.gradClip = 0.5
        self.dataStoreLen = 10000
        self.trainStep = 10

class Node:
    def __init__(self, point, parent = None):
        self.point = point
        self.parent = parent

class MapInfo:
    def __init__(self):
        self.mapInfo = self.GenerateGridMap()

    def GeneralObstacles(self):
        obstacles = []
        for _ in range(OBSTACLE_NUM):
            circle = [random.randint(OBSTACLE_RADIUS, MAP_LENGTH - OBSTACLE_RADIUS), random.randint(OBSTACLE_RADIUS, MAP_HEIGHT - OBSTACLE_RADIUS)]
            while numpy.linalg.norm(numpy.array(START) - numpy.array(circle)) < OBSTACLE_RADIUS or numpy.linalg.norm(numpy.array(END) - numpy.array(circle)) < OBSTACLE_RADIUS:
                circle = [random.randint(OBSTACLE_RADIUS, MAP_LENGTH - OBSTACLE_RADIUS), random.randint(OBSTACLE_RADIUS, MAP_HEIGHT - OBSTACLE_RADIUS)]
            obstacles.append(circle)
        return obstacles

    def CheckCollision(self, node):
        point = numpy.array(node.point)
        if self.mapInfo[int(point[0] / MAP_RESOLUTION)][int(point[1] / MAP_RESOLUTION)] == COLLISION:
            return True
        return False

    def GenerateGridMap(self, designObstacles=None):
        if designObstacles is not None:
            obstacles = designObstacles
        else:
            obstacles = self.GeneralObstacles()

        mapInfo = numpy.zeros((int(MAP_LENGTH / MAP_RESOLUTION + 1), int(MAP_HEIGHT / MAP_RESOLUTION + 1)))     # 可以触碰边界
        for obstacle in obstacles:
            circleCenter = [int(obstacle[0] / MAP_RESOLUTION), int(obstacle[1] / MAP_RESOLUTION)]
            circleRadius = int(OBSTACLE_RADIUS / MAP_RESOLUTION)

            for x in range(circleCenter[0] - circleRadius, circleCenter[0] + circleRadius + 1):
                for y in range(circleCenter[1] - circleRadius, circleCenter[1] + circleRadius + 1):
                    if numpy.linalg.norm(numpy.array(circleCenter) - numpy.array([x, y])) <= circleRadius:
                        mapInfo[x][y] = COLLISION
        return mapInfo

    def calDistance(self, nodes):
        node = nodes[-1]
        distance = 0
        while node.parent is not None:
            distance += numpy.linalg.norm(numpy.array(node.point) - numpy.array(node.parent.point))
            node = node.parent
        return distance

################################################################################

# PPONetWork
class StateFeatureNetWork(torch.nn.Module):
    def __init__(self, device) -> None:
        super(StateFeatureNetWork,self).__init__()
        self.device = device
        self.netWork = torch.nn.Sequential(
            torch.nn.Conv2d(5, 16, 5, 1, 1),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(16, 32, 5, 1, 1),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32, 32, 5, 1, 1),
            torch.nn.MaxPool2d(2),
            torch.nn.Flatten(),
            torch.nn.Linear(128, 20)
        )
        self.netWork.to(self.device)

    def forward(self, state):
        stateFeature = self.netWork(state)
        return stateFeature

class ActorNetWorkMu(torch.nn.Module):
    def __init__(self, device) -> None:
        super(ActorNetWorkMu,self).__init__()
        self.device = device
        self.netWork = torch.nn.Sequential(
            torch.nn.Linear(20, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1),
            torch.nn.Tanh()
        )
        self.netWork.to(self.device)

    def forward(self, stateFeature):
        Mu = self.netWork(stateFeature) * 180 + 180
        return Mu

class ActorNetWorkSigma(torch.nn.Module):
    def __init__(self, device) -> None:
        super(ActorNetWorkSigma,self).__init__()
        self.device = device
        self.netWork = torch.nn.Sequential(
            torch.nn.Linear(20, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1),
            torch.nn.Softplus()
        )
        self.netWork.to(self.device)

    def forward(self, stateFeature):
        Sigma = self.netWork(stateFeature)
        return Sigma

class ValueNetWork(torch.nn.Module):
    def __init__(self, device) -> None:
        super(ValueNetWork,self).__init__()
        self.device = device
        self.netWork = torch.nn.Sequential(
            torch.nn.Linear(20, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1)
        )
        self.netWork.to(self.device)

    def forward(self, stateFeature):
        value = self.netWork(stateFeature)
        return value

class PPONetWork():
    def __init__(self, device):
        self.StateFeatureModel = StateFeatureNetWork(device)
        self.NewActorMuModel = ActorNetWorkMu(device)
        self.NewActorSigmaModel = ActorNetWorkSigma(device)
        self.OldActorMuModel = ActorNetWorkMu(device)
        self.OldActorSigmaModel = ActorNetWorkSigma(device)
        self.ValueModel = ValueNetWork(device)

        self.hyperParameters = HyperParameters()
        self.actorOptimizer = torch.optim.Adam(list(self.NewActorMuModel.parameters()) + list(self.NewActorSigmaModel.parameters()), lr=self.hyperParameters.actorLr)
        self.valueOptimizer = torch.optim.Adam(list(self.StateFeatureModel.parameters()) + list(self.ValueModel.parameters()), lr=self.hyperParameters.criticLr)
        self.lossFunction = torch.nn.MSELoss()
        self.dataStore = []
        self.device = device
        self.initialize()
        self.InitialParameter()

    def initialize(self):
        self.reward = []
        self.actorLoss = []
        self.criticLoss = []

    def InitialParameter(self):
        self.NewActorMuModel.load_state_dict(self.OldActorMuModel.state_dict())
        self.NewActorSigmaModel.load_state_dict(self.OldActorSigmaModel.state_dict())

    def saveModel(self, path):
        torch.save({
            'stateFeatureModel_state_dict': self.StateFeatureModel.state_dict(),
            'ValueModel_state_dict': self.ValueModel.state_dict(),
            'NewActorMuModel_state_dict': self.NewActorMuModel.state_dict(),
            'OldActorMuModel_state_dict': self.OldActorMuModel.state_dict(),
            'NewActorSigmaModel_state_dict': self.NewActorSigmaModel.state_dict(),
            'OldActorSigmaModel_state_dict': self.OldActorSigmaModel.state_dict(),
        }, path)

    def loadModel(self, path):
        checkpoint = torch.load(path, weights_only=True)
        self.StateFeatureModel.load_state_dict(checkpoint['stateFeatureModel_state_dict'])
        self.ValueModel.load_state_dict(checkpoint['ValueModel_state_dict'])
        self.NewActorMuModel.load_state_dict(checkpoint['NewActorMuModel_state_dict'])
        self.OldActorMuModel.load_state_dict(checkpoint['OldActorMuModel_state_dict'])
        self.NewActorSigmaModel.load_state_dict(checkpoint['NewActorSigmaModel_state_dict'])
        self.OldActorSigmaModel.load_state_dict(checkpoint['OldActorSigmaModel_state_dict'])

        self.StateFeatureModel.eval()
        self.ValueModel.eval()
        self.NewActorMuModel.eval()
        self.NewActorSigmaModel.eval()
        self.NewActorSigmaModel.eval()
        self.OldActorSigmaModel.eval()

    def OldActorForward(self, state):
        state = torch.tensor(state, dtype=torch.float32).reshape(-1,5,SCAN_RANGE+1,SCAN_RANGE+1).to(self.device)
        stateFeature = self.StateFeatureModel(state)
        Mu, Sigma = self.OldActorMuModel(stateFeature), self.OldActorSigmaModel(stateFeature) + 1e-5
        distribution = torch.distributions.Normal(Mu, Sigma)
        action = distribution.sample().clamp(0, 360).reshape(-1, 1)
        logProb = distribution.log_prob(action).reshape(-1, 1)
        return action, logProb

    def NewActorForward(self, state, action):
        stateFeature = self.StateFeatureModel(state)
        Mu, Sigma = self.NewActorMuModel(stateFeature), self.NewActorSigmaModel(stateFeature) + 1e-5
        distribution = torch.distributions.Normal(Mu, Sigma)
        logProb = distribution.log_prob(action).reshape(-1, 1)
        return logProb

    def ValueForward(self, state):
        stateFeature = self.StateFeatureModel(state)
        value = self.ValueModel(stateFeature)
        return value
    
    def UpdateTargetNetWork(self, TargetNetWork, NetWork):
        TargetNetWork.load_state_dict(NetWork.state_dict())

    def PPOTrain(self):
        oldState = torch.tensor(numpy.array([data[0] for data in self.dataStore]), dtype=torch.float32).reshape(-1,5,SCAN_RANGE+1,SCAN_RANGE+1).to(self.device)
        oldAction = torch.tensor(numpy.array([data[1] for data in self.dataStore]), dtype=torch.float32).reshape(-1,1).to(self.device)
        reward = torch.tensor(numpy.array([data[2] for data in self.dataStore]), dtype=torch.float32).reshape(-1,1).to(self.device)
        oldState_ = torch.tensor(numpy.array([data[3] for data in self.dataStore]), dtype=torch.float32).reshape(-1,5,SCAN_RANGE+1,SCAN_RANGE+1).to(self.device)
        oldActionLogProb = torch.tensor([data[4] for data in self.dataStore], dtype=torch.float32).reshape(-1,1).to(self.device)

        reward = (reward - reward.mean()) / (reward.std() + 1e-5)
        with torch.no_grad():
            targetValue = reward + self.hyperParameters.gamma * self.ValueForward(oldState_)
        advantage = (targetValue - self.ValueForward(oldState)).detach()

        for _ in range(self.hyperParameters.trainStep):
            for index in BatchSampler(SubsetRandomSampler(range(len(self.dataStore))), 64, False):
                newActionLogProb = self.NewActorForward(oldState[index], oldAction[index])
                ratio = torch.exp(newActionLogProb - oldActionLogProb[index])

                L1 = ratio * advantage[index]
                L2 = torch.clamp(ratio, 1 - self.hyperParameters.gradEpsilon, 1 + self.hyperParameters.gradEpsilon) * advantage[index]
                ActorLoss = -torch.min(L1, L2).mean()
                self.actorOptimizer.zero_grad()
                ActorLoss.backward()
                torch.nn.utils.clip_grad_norm_(self.NewActorMuModel.parameters(), self.hyperParameters.gradClip)
                torch.nn.utils.clip_grad_norm_(self.NewActorSigmaModel.parameters(), self.hyperParameters.gradClip)
                self.actorOptimizer.step()
                self.actorLoss.append(ActorLoss.item())

                ValueLoss = self.lossFunction(self.ValueForward(oldState[index]), targetValue[index])
                self.valueOptimizer.zero_grad()
                ValueLoss.backward()
                torch.nn.utils.clip_grad_norm_(self.ValueModel.parameters(), self.hyperParameters.gradClip)
                self.valueOptimizer.step()
                self.criticLoss.append(ValueLoss.item())

        self.UpdateTargetNetWork(self.OldActorMuModel, self.NewActorMuModel)
        self.UpdateTargetNetWork(self.OldActorSigmaModel, self.NewActorSigmaModel)
        self.dataStore.clear()

    def play(self, environment, epoch):
        environment.reset()
        state = environment.getState()
        over, printFlag = False, False
        while not over:
            action, logProb = self.OldActorForward(state)
            state_, reward, over = environment.step(action)
            self.dataStore.append((state, action.item(), reward, state_, logProb))
            if len(self.dataStore) >= self.hyperParameters.dataStoreLen:
                self.PPOTrain()
                printFlag = True

            state = state_
            self.reward.append(reward)

        sumReward = sum(self.reward)
        if printFlag == True:
            print(f'Epoch: {epoch}, ActorLoss: {sum(self.actorLoss)}, CriticLoss: {sum(self.criticLoss)}, Reward: {sumReward}')

        self.initialize()
        return sumReward

class Environment():
    def __init__(self):
        self.mapInfo = MapInfo()
        self.nodes = [Node(START)]

    def reset(self):
        self.nodes = [Node(START)]

    def step(self, action):
        over = False
        if numpy.linalg.norm(numpy.array(self.nodes[-1].point) - numpy.array(END)) < STEP_SIZE:
            newNode = Node(list(END), self.nodes[-1])
            self.nodes.append(newNode)
            over = True
        else:
            newPoint = numpy.array(self.nodes[-1].point) + numpy.array([STEP_SIZE * math.cos(action.item() / 180 * math.pi), STEP_SIZE * math.sin(action.item() / 180 * math.pi)])
            newNode = Node(list(newPoint), self.nodes[-1])
            self.nodes.append(newNode)
            if newPoint[0] >= 0 and newPoint[0] <= MAP_LENGTH and newPoint[1] >= 0 and newPoint[1] <= MAP_HEIGHT:
                if self.mapInfo.CheckCollision(newNode):
                    over = True
            else:
                over = True
        reward = self.getReward()
        state = self.getState()

        if len(self.nodes) > MAX_NODES:
            over = True
        return state, reward, over

    def render(self, path=None):
        matplotlib.pyplot.figure(figsize=(10, 10))
        finalPath = []
        node = self.nodes[-1]
        while node.parent is not None:
            finalPath.append(node.point)
            node = node.parent
        finalPath.reverse()

        for node in self.nodes:
            if node.parent is not None and node not in finalPath:
                matplotlib.pyplot.plot([int(node.point[0] / MAP_RESOLUTION), int(node.parent.point[0] / MAP_RESOLUTION)], [int(node.point[1] / MAP_RESOLUTION), int(node.parent.point[1] / MAP_RESOLUTION)], 'r-')

        for item in range(len(finalPath) - 1):
            matplotlib.pyplot.plot([int(finalPath[item][0] / MAP_RESOLUTION), int(finalPath[item + 1][0] / MAP_RESOLUTION)], [int(finalPath[item][1] / MAP_RESOLUTION), int(finalPath[item + 1][1] / MAP_RESOLUTION)], 'k-')

        for row in range(len(self.mapInfo.mapInfo)):
            for column in range(len(self.mapInfo.mapInfo[row])):
                if self.mapInfo.mapInfo[row][column] == COLLISION:
                    # 对每一个障碍物格子填色
                    rectangle = matplotlib.pyplot.Rectangle((row, column), MAP_RESOLUTION, MAP_RESOLUTION, edgecolor='blue', facecolor='blue')
                    matplotlib.pyplot.gca().add_patch(rectangle)
            
        matplotlib.pyplot.plot(int(START[0] / MAP_RESOLUTION), int(START[1] / MAP_RESOLUTION), 'go')
        matplotlib.pyplot.plot(int(END[0] / MAP_RESOLUTION), int(END[1] / MAP_RESOLUTION), 'go')
        matplotlib.pyplot.xlim(0, int(MAP_LENGTH / MAP_RESOLUTION + 1))
        matplotlib.pyplot.ylim(0, int(MAP_HEIGHT / MAP_RESOLUTION + 1))
        matplotlib.pyplot.gca().set_aspect('equal', adjustable='box')
        if path is not None:
            matplotlib.pyplot.savefig(path, dpi=1200)
        else:
            matplotlib.pyplot.show()

    def getState(self):
        positionInfo = [int(item / MAP_RESOLUTION) for item in self.nodes[-1].point]
        state = [[0 for _ in range(SCAN_RANGE + 1)] for _ in range(SCAN_RANGE + 1)]

        for x in range(SCAN_RANGE + 1):
            for y in range(SCAN_RANGE + 1):
                map_x = x+(positionInfo[0] - SCAN_RANGE//2)
                map_y = y+(positionInfo[1] - SCAN_RANGE//2)
                if map_x >= 0 and map_x <= int(MAP_LENGTH / MAP_RESOLUTION) and map_y >= 0 and map_y <= int(MAP_HEIGHT / MAP_RESOLUTION) :
                    state[x][y] = self.mapInfo.mapInfo[map_x][map_y]
                else:
                    state[x][y] = OUT_MAP
        state[SCAN_RANGE//2][SCAN_RANGE//2] = ROBOT
        state = numpy.array(state)
        selfMask1 = numpy.full((SCAN_RANGE + 1, SCAN_RANGE + 1), positionInfo[0])
        selfMask2 = numpy.full((SCAN_RANGE + 1, SCAN_RANGE + 1), positionInfo[1])
        endMask1 = numpy.full((SCAN_RANGE + 1, SCAN_RANGE + 1), END[0])
        endMask2 = numpy.full((SCAN_RANGE + 1, SCAN_RANGE + 1), END[1])
        finalState = numpy.dstack((state, selfMask1, selfMask2, endMask1, endMask2)).transpose((2, 0, 1))
        return finalState

    def getReward(self):
        if self.nodes[-1].point == END:
            reward = reward = (MAX_NODES - len(self.nodes)) * 0.1 + 50
        elif (self.nodes[-1].point[0] >= 0 and self.nodes[-1].point[0] <= 50 and self.nodes[-1].point[1] >= 0 and self.nodes[-1].point[1] <= 50) and self.mapInfo.CheckCollision(self.nodes[-1]):
            reward = -50
        else:
            oldDistance = numpy.linalg.norm(numpy.array(self.nodes[-2].point) - numpy.array(END))
            newDistance = numpy.linalg.norm(numpy.array(self.nodes[-1].point) - numpy.array(END))
            moveRate = (oldDistance - newDistance) / STEP_SIZE
            reward = ALPHA * moveRate + 0.05

        return reward

def modelTrain():
    PPONet = PPONetWork("cuda")
    writer = SummaryWriter('C:\\Users\\60520\\Desktop\\RL-learning\\Log\\PP-PPO')

    for mapIndex in range(10):
        environment = Environment()
        environment.render('C:\\Users\\60520\\Desktop\\RL-learning\\PathPlanning\\' + 'PPO-figure-' + str(0) + '.png')
        for epoch in range(50000):
            writer.add_scalar('reward-epoch', PPONet.play(environment, epoch), epoch)
            if (epoch + 1) % 5000 == 0:
                PPONet.saveModel('C:\\Users\\60520\\Desktop\\RL-learning\\PathPlanning\\' + 'PPO-model-' + str(epoch + 1) + '.pth')
                environment.render('C:\\Users\\60520\\Desktop\\RL-learning\\PathPlanning\\' + 'PPO-figure-' + str(epoch + 1) + '.png')

    writer.close()

def modelTest():
    PPONet = PPONetWork("cuda")
    environment = Environment()
    modelName = ''
    PPONet.loadModel('C:\\Users\\60520\\Desktop\\RL-learning\\PathPlanning\\' + modelName)

    PPONet.play(environment, 0)
    environment.render()

# main
def main():
    typeParameter = 0
    if typeParameter == 0:
        modelTrain()
    if typeParameter == 1:
        modelTest()

    return

if __name__ == '__main__':
    main()