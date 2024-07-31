import math
import numpy
import torch
import random
import matplotlib.pyplot
from torch.utils.tensorboard import SummaryWriter

# parameters
START = [2, 2]          # 起点坐标
END = [48, 48]          # 终点坐标
MAX_NODES = 200         # 节点最大数量
STEP_SIZE = 0.5         # 步长
OBSTACLE_NUM = 9        # 障碍物数量
OBSTACLE_RADIUS = 4     # 障碍物半径
MAP_LENGTH = 50         # 地图长度
MAP_HEIGHT = 50         # 地图宽度
MAP_RESOLUTION = 0.5    # 地图分辨率
SCAN_RANGE = 31         # 智能体扫描范围
ALPHA = 2               # 奖励权重

# enumeration value
COLLISION = 127
ROBOT = 255
OUT_MAP = -1

class HyperParameters():
    def __init__(self):
        self.actorLr = 1e-4
        self.criticLr = 1e-4
        self.gamma = 0.99
        self.epsilon = 0.999
        self.dataStoreLen = 100000
        self.initStoreLen = 500

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

# ACNetWork
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

class ActorNetWork(torch.nn.Module):
    def __init__(self, device) -> None:
        super(ActorNetWork,self).__init__()
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
        action = self.netWork(stateFeature) * 180 + 180
        return action

class CriticNetWork(torch.nn.Module):
    def __init__(self, device) -> None:
        super(CriticNetWork,self).__init__()
        self.device = device
        self.netWork = torch.nn.Sequential(
            torch.nn.Linear(21, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1)
        )
        self.netWork.to(self.device)

    def forward(self, stateFeature, action):
        tensor = torch.cat((stateFeature, action), 1)
        value = self.netWork(tensor)
        return value

class ACNetWork():
    def __init__(self, device):
        self.StateFeatureModel = StateFeatureNetWork(device)
        self.ActorModel = ActorNetWork(device)
        self.TargetActorModel = ActorNetWork(device)
        self.CriticModel = CriticNetWork(device)
        self.TargetCriticModel = CriticNetWork(device)

        self.hyperParameters = HyperParameters()
        self.actorOptimizer = torch.optim.Adam(self.ActorModel.parameters(), lr=self.hyperParameters.actorLr)
        self.criticOptimizer = torch.optim.Adam(list(self.StateFeatureModel.parameters()) + list(self.CriticModel.parameters()), lr=self.hyperParameters.criticLr)
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
        self.TargetActorModel.load_state_dict(self.ActorModel.state_dict())
        self.TargetCriticModel.load_state_dict(self.CriticModel.state_dict())

    def saveModel(self, path):
        torch.save({
            'stateFeatureModel_state_dict': self.StateFeatureModel.state_dict(),
            'CriticModel_state_dict': self.CriticModel.state_dict(),
            'TargetCriticModel_state_dict': self.TargetCriticModel.state_dict(),
            'ActorModel_state_dict': self.ActorModel.state_dict(),
            'TargetActorModel_state_dict': self.TargetActorModel.state_dict(),
        }, path)

    def loadModel(self, path):
        checkpoint = torch.load(path, weights_only=True)
        self.StateFeatureModel.load_state_dict(checkpoint['stateFeatureModel_state_dict'])
        self.CriticModel.load_state_dict(checkpoint['CriticModel_state_dict'])
        self.TargetCriticModel.load_state_dict(checkpoint['TargetCriticModel_state_dict'])
        self.ActorModel.load_state_dict(checkpoint['ActorModel_state_dict'])
        self.TargetActorModel.load_state_dict(checkpoint['TargetActorModel_state_dict'])

        self.StateFeatureModel.eval()
        self.CriticModel.eval()
        self.TargetCriticModel.eval()
        self.ActorModel.eval()
        self.TargetActorModel.eval()

    def actorForward(self, state):
        stateFeature = self.StateFeatureModel(state)
        action = self.ActorModel(stateFeature)
        return action
    
    def criticForward(self, state, action):
        stateFeature = self.StateFeatureModel(state)
        value = self.CriticModel(stateFeature, action)
        return value
    
    def targetActorForward(self, state):
        stateFeature = self.StateFeatureModel(state)
        action = self.TargetActorModel(stateFeature)
        return action

    def targetCriticForward(self, state, action):
        stateFeature = self.StateFeatureModel(state)
        value = self.TargetCriticModel(stateFeature, action)
        return value
    
    def getAction(self, state):
        if random.random() < self.hyperParameters.epsilon:
            action = torch.tensor(numpy.random.uniform(0, 360, 1), dtype=torch.float32).reshape(-1,1).to(self.device)
        else:
            state = torch.tensor(state, dtype=torch.float32).reshape(-1,5,SCAN_RANGE+1,SCAN_RANGE+1).to(self.device)
            action = self.actorForward(state)
        return action
    
    def UpdateTargetNetWork(self, TargetNetWork, NetWork):
        for targetParam, param in zip(TargetNetWork.parameters(), NetWork.parameters()):
            targetParam.data.copy_(0.005 * param.data + (1 - 0.005) * targetParam.data)

    def DPGTrain(self):
        chosenData = random.sample(self.dataStore, 64)
        state = torch.tensor(numpy.array([data[0] for data in chosenData]), dtype=torch.float32).reshape(-1,5,SCAN_RANGE+1,SCAN_RANGE+1).to(self.device)
        action = torch.tensor(numpy.array([data[1] for data in chosenData]), dtype=torch.float32).reshape(-1,1).to(self.device)
        reward = torch.tensor(numpy.array([data[2] for data in chosenData]), dtype=torch.float32).reshape(-1,1).to(self.device)
        state_ = torch.tensor(numpy.array([data[3] for data in chosenData]), dtype=torch.float32).reshape(-1,5,SCAN_RANGE+1,SCAN_RANGE+1).to(self.device)
        over = torch.tensor(numpy.array([data[4] for data in chosenData]), dtype=torch.float32).reshape(-1,1).to(self.device)

        QValue = self.criticForward(state, action)

        action_ = self.targetActorForward(state_)
        QValue_ = self.targetCriticForward(state_, action_.reshape(-1, 1).to(self.device))
        Target = reward + self.hyperParameters.gamma * QValue_ * (1 - over)
        CriticLoss = self.lossFunction(QValue, Target)

        self.criticOptimizer.zero_grad()
        CriticLoss.backward()
        self.criticOptimizer.step()
        self.criticLoss.append(CriticLoss.item())

        actionPre = self.actorForward(state)
        QValue = self.criticForward(state, actionPre.reshape(-1, 1).to(self.device))
        ActorLoss = -(QValue).mean()

        self.actorOptimizer.zero_grad()
        ActorLoss.backward()
        self.actorOptimizer.step()
        self.actorLoss.append(ActorLoss.item())

        self.UpdateTargetNetWork(self.TargetActorModel, self.ActorModel)
        self.UpdateTargetNetWork(self.TargetCriticModel, self.CriticModel)
    
    def getData(self, environment):
        while len(self.dataStore) < self.hyperParameters.initStoreLen:
            environment.reset()
            state = environment.getState()
            over = False
            while not over:
                action = self.getAction(state).detach().cpu().numpy()
                state_, reward, over = environment.step(action)
                self.dataStore.append((state, action, reward, state_, over))
                state = state_

    def play(self, environment, epoch):
        environment.reset()
        state = environment.getState()
        over = False
        while not over:
            action = self.getAction(state).detach().cpu().numpy()
            state_, reward, over = environment.step(action)
            self.dataStore.append((state, action, reward, state_, over))
            state = state_
            self.reward.append(reward)

        sumReaward = sum(self.reward)
        while len(self.dataStore) > self.hyperParameters.dataStoreLen:
            self.dataStore.pop(0)
        
        print(f'Epoch: {epoch}, ActorLoss: {sum(self.actorLoss)}, CriticLoss: {sum(self.criticLoss)}, sumReward: {sum(self.reward)}')
        self.initialize()
        return sumReaward

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
            reward = 50
        elif (self.nodes[-1].point[0] >= 0 and self.nodes[-1].point[0] <= 50 and self.nodes[-1].point[1] >= 0 and self.nodes[-1].point[1] <= 50) and self.mapInfo.CheckCollision(self.nodes[-1]):
            reward = -50
        else:
            oldDistance = numpy.linalg.norm(numpy.array(self.nodes[-2].point) - numpy.array(END))
            newDistance = numpy.linalg.norm(numpy.array(self.nodes[-1].point) - numpy.array(END))
            moveRate = (oldDistance - newDistance) / STEP_SIZE
            reward = ALPHA * moveRate + 0.05

        return reward

def modelTrain():
    DPGNet = ACNetWork("cuda")
    environment = Environment()
    writer = SummaryWriter('C:\\Users\\60520\\Desktop\\RL-learning\\Log\\PP-DPG')
    DPGNet.getData(environment)

    for epoch in range(100000):
        for index in range(10):
            DPGNet.DPGTrain()
        DPGNet.hyperParameters.epsilon = max(DPGNet.hyperParameters.epsilon * 0.9998, 0.05)
        writer.add_scalar('reward-epoch', DPGNet.play(environment, epoch), epoch)
        if (epoch + 1) % 5000 == 0:
            DPGNet.saveModel('C:\\Users\\60520\\Desktop\\RL-learning\\PathPlanning\\' + 'model-' + str(epoch + 1) + '.pth')
            environment.render('C:\\Users\\60520\\Desktop\\RL-learning\\PathPlanning\\' + 'figure-' + str(epoch + 1) + '.png')

def modelTest():
    DPGNet = ACNetWork("cuda")
    environment = Environment()
    modelName = 'model-20000.pth'
    DPGNet.loadModel('C:\\Users\\60520\\Desktop\\RL-learning\\PathPlanning\\' + modelName)

    DPGNet.hyperParameters.epsilon = 0
    DPGNet.play(environment, 0)
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