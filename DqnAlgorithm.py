import math
import copy
import numpy
import random
import datetime
import torch.nn
import matplotlib.pyplot

class Node:
    def __init__(self, point, parent = None):
        self.point = point
        self.parent = parent

# parameters
START = [1, 1]          # 起点坐标
END = [50, 50]          # 终点坐标
MAX_NODES = 2000        # 节点最大数量
STEP_SIZE = 0.5         # 步长
OBSTACLE_NUM = 5        # 障碍物数量
OBSTACLE_RADIUS = 5     # 障碍物半径
MAP_LENGTH = 50         # 地图长度
MAP_HEIGHT = 50         # 地图宽度
MAP_RESOLUTION = 0.1    # 地图分辨率
EPOCHS = 1000           # 训练轮数
EPSILON = 0.99          # epsilon-greedy

# general obstacles
def GeneralObstacles():
    obstacles = []
    for _ in range(OBSTACLE_NUM):
        circle = [random.randint(OBSTACLE_RADIUS, MAP_LENGTH - OBSTACLE_RADIUS), random.randint(OBSTACLE_RADIUS, MAP_HEIGHT - OBSTACLE_RADIUS)]
        while numpy.linalg.norm(numpy.array(START) - numpy.array(circle)) < OBSTACLE_RADIUS or numpy.linalg.norm(numpy.array(END) - numpy.array(circle)) < OBSTACLE_RADIUS:
            circle = [random.randint(OBSTACLE_RADIUS, MAP_LENGTH - OBSTACLE_RADIUS), random.randint(OBSTACLE_RADIUS, MAP_HEIGHT - OBSTACLE_RADIUS)]
        obstacles.append(circle)
    return obstacles

# check collision
def CheckCollision(node, mapInfo):
    point = numpy.array(node.point)
    if mapInfo[int(point[0] / MAP_RESOLUTION)][int(point[1] / MAP_RESOLUTION)] == 1:
        return True
    return False

# general grid map
def GenerateGridMap():
    mapInfo = numpy.zeros((int(MAP_LENGTH / MAP_RESOLUTION + 1), int(MAP_HEIGHT / MAP_RESOLUTION + 1)))     # 可以触碰边界
    obstacles = GeneralObstacles()
    for obstacle in obstacles:
        circleCenter = [int(obstacle[0] / MAP_RESOLUTION), int(obstacle[1] / MAP_RESOLUTION)]
        circleRadius = int(OBSTACLE_RADIUS / MAP_RESOLUTION)

        for x in range(circleCenter[0] - circleRadius, circleCenter[0] + circleRadius + 1):
            for y in range(circleCenter[1] - circleRadius, circleCenter[1] + circleRadius + 1):
                if numpy.linalg.norm(numpy.array(circleCenter) - numpy.array([x, y])) <= circleRadius:
                    mapInfo[x][y] = 255
    return mapInfo

def calDistance(nodes):
    node = nodes[-1]
    distance = 0
    while node.parent is not None:
        distance += numpy.linalg.norm(numpy.array(node.point) - numpy.array(node.parent.point))
        node = node.parent
    return distance

################################################################################

# ACNetWork
class ACNetWork(torch.nn.Module):
    def __init__(self, device) -> None:
        super(ACNetWork,self).__init__()
        ActorLayer = [
            # 尺寸1024之内的都会被池化为1并映射为1维标量
            torch.nn.Conv2d(1, 32, 5, 1, 2),
            torch.nn.AvgPool2d(5, ceil_mode=True),
            torch.nn.Conv2d(32, 32, 5, 1, 2),
            torch.nn.AvgPool2d(5, ceil_mode=True),
            torch.nn.Conv2d(32, 64, 5, 1, 2),
            torch.nn.AvgPool2d(5, ceil_mode=True),
            torch.nn.Conv2d(64, 64, 5, 1, 2),
            torch.nn.AvgPool2d(5, ceil_mode=True),
            torch.nn.Conv2d(64, 64, 5, 1, 2),
            torch.nn.AvgPool2d(5, ceil_mode=True),
            torch.nn.Flatten(),
            torch.nn.Linear(64, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 1)
        ]
        ActionLayer = [
            torch.nn.Linear(3, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1)
        ]

        CriticStateLayer = [
            torch.nn.Conv2d(1, 32, 5, 1, 2),
            torch.nn.AvgPool2d(5, ceil_mode=True),
            torch.nn.Conv2d(32, 32, 5, 1, 2),
            torch.nn.AvgPool2d(5, ceil_mode=True),
            torch.nn.Conv2d(32, 64, 5, 1, 2),
            torch.nn.AvgPool2d(5, ceil_mode=True),
            torch.nn.Conv2d(64, 64, 5, 1, 2),
            torch.nn.AvgPool2d(5, ceil_mode=True),
            torch.nn.Conv2d(64, 64, 5, 1, 2),
            torch.nn.AvgPool2d(5, ceil_mode=True),
            torch.nn.Flatten(),
            torch.nn.Linear(64, 1)
        ]
        CriticActionLayer = [
            torch.nn.Linear(2, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1)
        ]

        self.ActorModel = torch.nn.Sequential(*ActorLayer)
        self.ActionModel = torch.nn.Sequential(*ActionLayer)
        self.CriticStateModel = torch.nn.Sequential(*CriticStateLayer)
        self.CriticActionModel = torch.nn.Sequential(*CriticActionLayer)
        self.device = device
        self.initialize()

    def standardizeState(self, degree):
        degree = degree % 360
        degree[degree < 0] += 360
        return degree

    def ActorForward(self, state, position):
        mapFeature = torch.tensor(self.ActorModel(state).item(), dtype=torch.float32).to(self.device)
        tensor = torch.cat((torch.stack([mapFeature]),position),0)
        return self.standardizeState(self.ActionModel(tensor))
    
    def CriticForward(self, state, action):
        stateFeature = torch.tensor(self.CriticStateModel(state).item(), dtype=torch.float32).to(self.device)
        tensor = torch.stack([stateFeature, action])
        return self.CriticActionModel(tensor)
    
    def initialize(self):
        self.ActorLoss = []
        self.CriticLoss = []

    def saveModel(self, path):
        torch.save(self.state_dict(), path)

    def loadModel(self, path):
        self.load_state_dict(torch.load(path))

# training
def Train(state, action, reward, nextState, over, ActorOptimizer, CriticOptimizer, lossFunction, ACNet):
    state, position = state[0].unsqueeze(0).unsqueeze(1), state[1]
    nextState, nextPosition = nextState[0].unsqueeze(0).unsqueeze(1), nextState[1]
    # train
    Qvalue = ACNet.CriticForward(state, action)
    with torch.no_grad():
        nextAction = torch.tensor(ACNet.ActorForward(nextState, nextPosition).item(), dtype=torch.float32).to(ACNet.device)
        nextQvalue = ACNet.CriticForward(nextState, nextAction)
        target = reward + 0.99 * nextQvalue * (1 - over)
    CriticLoss = lossFunction(Qvalue, target)
    ACNet.CriticLoss.append(CriticLoss)
    # 更新评论家网络
    CriticOptimizer.zero_grad()
    CriticLoss.backward()
    CriticOptimizer.step()

    # 计算演员网络的损失
    actionPred = torch.tensor(ACNet.ActorForward(state, position).item(), dtype=torch.float32).to(ACNet.device)
    Qvalues = ACNet.CriticForward(state, actionPred)
    ActorLoss = -Qvalues.mean()
    ACNet.ActorLoss.append(ActorLoss)
    # 更新演员网络
    ActorOptimizer.zero_grad()
    ActorLoss.backward()
    ActorOptimizer.step()

# get state
def GetState(nodes, mapInfo, ACNet):
    positionInfo = [int(item / MAP_RESOLUTION) for item in nodes[-1].point]
    enviroment = copy.deepcopy(mapInfo)
    enviroment[positionInfo[0]][positionInfo[1]] = 100
    state = [torch.tensor(enviroment, dtype=torch.float32).to(ACNet.device), torch.tensor(positionInfo, dtype=torch.float32).to(ACNet.device)]
    # state是一个列表,二维张量,具体长度和地图尺寸相关
    return state

# get action
def GetAction(state, ACNet):
    state, position = state[0].unsqueeze(0).unsqueeze(1), state[1]
    action = torch.tensor(ACNet.ActorForward(state, position).item(), dtype=torch.float32).to(ACNet.device)
    # action是一个一维张量,长度为1
    return action

# get reward
def GetReward(nodes, mapInfo):
    if nodes[-1].point == END:
        reward = (MAX_NODES - len(nodes)) * 0.001 + 1
    elif CheckCollision(nodes[-1], mapInfo):
        reward = -1
    else:
        if len(nodes) != 1:
            oldDistance = numpy.linalg.norm(numpy.array(nodes[-2].point) - numpy.array(END))
            newDistance = numpy.linalg.norm(numpy.array(nodes[-1].point) - numpy.array(END))
            moveRate = (oldDistance - newDistance) / STEP_SIZE
            reward = moveRate + 0.0005
        else:
            reward = -1
    return reward

# step
def Step(nodes, action, mapInfo):
    over = False
    if numpy.linalg.norm(numpy.array(nodes[-1].point) - numpy.array(END)) < STEP_SIZE:
        newNode = Node(list(END), nodes[-1])
        nodes.append(newNode)
        over = True
    else:
        newPoint = numpy.array(nodes[-1].point) + numpy.array([STEP_SIZE * math.cos(action.item() / 180 * math.pi), STEP_SIZE * math.sin(action.item() / 180 * math.pi)])
        if newPoint[0] >= 0 and newPoint[0] <= 50 and newPoint[1] >= 0 and newPoint[1] <= 50:
            newNode = Node(list(newPoint), nodes[-1])
            if CheckCollision(newNode, mapInfo):
                over = True
                nodes.append(newNode)
            else:
                nodes.append(newNode)
        else:
            over = True
    reward = GetReward(nodes, mapInfo)

    if len(nodes) > MAX_NODES:
        over = True
    return nodes, reward, over

# reinforcement learning
def ReinforcementLearning(device):
    ACNet = ACNetWork(device)
    ACNet.to(device)
    ActorOptimizer = torch.optim.Adam(ACNet.ActorModel.parameters(),lr=0.01)
    CriticOptimizer = torch.optim.Adam(list(ACNet.CriticStateModel.parameters()) + list(ACNet.CriticActionModel.parameters()),lr=0.01)
    lossFunction = torch.nn.MSELoss()

    mapInfo = GenerateGridMap()
    
    for epoch in range(EPOCHS):
        # on-policy training
        for _ in range(100):
            nodes = [Node(START)]
            state = GetState(nodes, mapInfo, ACNet)
            over = False
            sumReward = 0
            while not over:
                action = GetAction(state, ACNet)
                nodes, reward, over = Step(nodes, action, mapInfo)
                nextState = GetState(nodes, mapInfo, ACNet)
                state = nextState
                sumReward += reward
                # state:张量, action:张量, reward:浮点数, nextState:张量, over:布尔值
                Train(state, action, reward, nextState, over, ActorOptimizer, CriticOptimizer, lossFunction, ACNet)
            text = f'epoch: {epoch}, trainNum: {_}, ActorLoss: {sum(ACNet.ActorLoss)}, CriticLoss: {sum(ACNet.CriticLoss)}, sumReward: {sumReward}, stepNum: {len(nodes)}'
            saveTrainText(text)
            ACNet.initialize()
        
        # save model
        if epoch % 10 == 0:
            ACNet.saveModel(f'model-{epoch}.pkl')

        # plan on this map
        Play(mapInfo, ACNet)
        
        # plan on new map
        newMapInfo = GenerateGridMap()
        Play(newMapInfo, ACNet)

# play
def Play(mapInfo, ACNet):
    nodes = [Node(START)]
    sumReward = 0
    for nodeNumber in range(MAX_NODES):
        state = GetState(nodes, mapInfo, ACNet)
        action = GetAction(state, ACNet)
        nodes, reward, over = Step(nodes, action, mapInfo)
        sumReward += reward
        if over:
            if nodes[-1].point == END:
                text = f'nodeNumber: {nodeNumber}, pathDistance: {calDistance(nodes)}, sumReward: {sumReward}'
                saveTrainText(text)
            else:
                text = f'collision or out of map and step number is {len(nodes)}, sumReward: {sumReward}'
                saveTrainText(text)
            break
    if not over:
        text = f'no path, sumReward: {sumReward}'
        saveTrainText(text)
    # DrawRRT(nodes, mapInfo)

# save train text
def saveTrainText(str):
    with open('train.txt', 'a') as f:
        f.write(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + '\n')
        f.write(str + '\n')
    print(str)

# draw
def DrawRRT(nodes, mapInfo):
    matplotlib.pyplot.figure(figsize=(10, 10))
    finalPath = []
    node = nodes[-1]
    while node.parent is not None:
        finalPath.append(node.point)
        node = node.parent
    finalPath.reverse()

    for node in nodes:
        if node.parent is not None and node not in finalPath:
            matplotlib.pyplot.plot([node.point[0], node.parent.point[0]], [node.point[1], node.parent.point[1]], 'r-')

    for item in range(len(finalPath) - 1):
        matplotlib.pyplot.plot([finalPath[item][0], finalPath[item + 1][0]], [finalPath[item][1], finalPath[item + 1][1]], 'k-')

    for row in range(len(mapInfo)):
        for column in range(len(mapInfo[row])):
            if mapInfo[row][column] == 1:
                rectangle = matplotlib.pyplot.Rectangle((row, column), MAP_RESOLUTION, MAP_RESOLUTION, edgecolor='blue', facecolor='blue')
                matplotlib.pyplot.gca().add_patch(rectangle)
        
    matplotlib.pyplot.plot(START[0], START[1], 'go')
    matplotlib.pyplot.plot(END[0], END[1], 'go')
    matplotlib.pyplot.xlim(0, 50)
    matplotlib.pyplot.ylim(0, 50)
    matplotlib.pyplot.gca().set_aspect('equal', adjustable='box')
    matplotlib.pyplot.show()

# main
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    ReinforcementLearning(device)

if __name__ == '__main__':
    main()