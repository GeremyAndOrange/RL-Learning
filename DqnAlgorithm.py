import math
import numpy
import random
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
                    mapInfo[x][y] = 1
    return mapInfo

def calDistance(nodes):
    node = nodes[-1]
    distance = 0
    while node.parent is not None:
        distance += numpy.linalg.norm(numpy.array(node.point) - numpy.array(node.parent.point))
        node = node.parent
    return distance

################################################################################

# parameters
EPSILON = 0.99          # greedy policy
EPOCHS = 1000           # 训练轮数

# ACNetWork
class ACNetWork(torch.nn.Module):
    def __init__(self) -> None:
        super(ACNetWork,self).__init__()
        ActorLayer = [
            # 尺寸1024之内的都会被池化为1
            torch.nn.Conv2d(1, 32, 5, 1, 2),
            torch.nn.MaxPool2d(5, ceil_mode=True),
            torch.nn.Conv2d(32, 32, 5, 1, 2),
            torch.nn.MaxPool2d(5, ceil_mode=True),
            torch.nn.Conv2d(32, 64, 5, 1, 2),
            torch.nn.MaxPool2d(5, ceil_mode=True),
            torch.nn.Conv2d(64, 64, 5, 1, 2),
            torch.nn.MaxPool2d(5, ceil_mode=True),
            torch.nn.Conv2d(64, 64, 5, 1, 2),
            torch.nn.MaxPool2d(5, ceil_mode=True),
            torch.nn.Flatten(),
            torch.nn.Linear(64, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 1),
            torch.nn.Softmax(1),
        ]
        CriticLayer = [
            # 尺寸1024之内的都会被池化为1
            torch.nn.Conv2d(1, 32, 5, 1, 2),
            torch.nn.MaxPool2d(5, ceil_mode=True),
            torch.nn.Conv2d(32, 32, 5, 1, 2),
            torch.nn.MaxPool2d(5, ceil_mode=True),
            torch.nn.Conv2d(32, 64, 5, 1, 2),
            torch.nn.MaxPool2d(5, ceil_mode=True),
            torch.nn.Conv2d(64, 64, 5, 1, 2),
            torch.nn.MaxPool2d(5, ceil_mode=True),
            torch.nn.Conv2d(64, 64, 5, 1, 2),
            torch.nn.MaxPool2d(5, ceil_mode=True),
            torch.nn.Flatten(),
            torch.nn.Linear(64, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 1),
            torch.nn.Softmax(1),
        ]
        self.ActorModel = torch.nn.Sequential(*ActorLayer)
        self.CriticModel = torch.nn.Sequential(*CriticLayer)
        self.initialize()

    def ActorForward(self, state):
        return self.ActorModel(state)
    
    def CriticForward(self, state, action):
        tensor = torch.cat((state, action), 2)
        return self.CriticModel(tensor)
    
    def initialize(self):
        self.ActorLoss = []
        self.CriticLoss = []

# training
def Train(dataPool, ACNet, ActorOptimizer, CriticOptimizer, lossFunction):
    trainData = random.sample(dataPool, 128)
    state = torch.stack([data[0] for data in trainData]).unsqueeze(1)
    action = torch.stack([data[1] for data in trainData]).unsqueeze(1).unsqueeze(1).unsqueeze(1).expand(128,1,1,state.size(3))
    nextState = torch.stack([data[3] for data in trainData]).unsqueeze(1)
    reward = torch.tensor(numpy.array([data[2] for data in trainData]), dtype=torch.float32).unsqueeze(1)
    over = torch.tensor(numpy.array([data[4] for data in trainData]), dtype=torch.int64).unsqueeze(1)

    Qvalues = ACNet.CriticForward(state, action)
    with torch.no_grad():
        nextAction = ACNet.ActorForward(nextState).unsqueeze(1).unsqueeze(1).expand(128,1,1,nextState.size(3))
        nextQvalues = ACNet.CriticForward(nextState, nextAction)
        targets = reward + 0.99 * nextQvalues * (1 - over)
    CriticLoss = lossFunction(Qvalues, targets)
    ACNet.CriticLoss.append(CriticLoss)
    # 更新评论家网络
    CriticOptimizer.zero_grad()
    CriticLoss.backward()
    CriticOptimizer.step()

    # 计算演员网络的损失
    actionPred = ACNet.ActorForward(state).unsqueeze(1).unsqueeze(1).expand(128,1,1,state.size(3))
    Qvalues = ACNet.CriticForward(state, actionPred)
    ActorLoss = -Qvalues.mean()
    ACNet.ActorLoss.append(ActorLoss)
    # 更新演员网络
    ActorOptimizer.zero_grad()
    ActorLoss.backward()
    ActorOptimizer.step()

# get state
def GetState(nodes, mapInfo):
    positionInfo = [int(item / MAP_RESOLUTION) for item in nodes[-1].point]
    mapInfo[positionInfo[0]][positionInfo[1]] = 100
    # state是一个二维张量,具体长度和地图尺寸相关
    state = torch.tensor(mapInfo, dtype=torch.float32)
    return state

# get action
def GetAction(state, ACNet):
    if EPSILON > numpy.random.rand():
        action = torch.tensor(random.random() * 360, dtype=torch.float32)
    else:
        state = state.unsqueeze(0).unsqueeze(1)
        action = torch.tensor(ACNet.ActorForward(state).item(), dtype=torch.float32)
    # action是一个一维张量,长度为1
    return action

# get reward
def GetReward(nodes, mapInfo):
    if nodes[-1].point == END:
        reward = (MAX_NODES - len(nodes)) * 1
    elif CheckCollision(nodes[-1], mapInfo):
        reward = -(MAX_NODES - len(nodes))
    else:
        if len(nodes) != 1:
            oldDistance = numpy.linalg.norm(numpy.array(nodes[-2].point) - numpy.array(END))
            newDistance = numpy.linalg.norm(numpy.array(nodes[-1].point) - numpy.array(END))
            moveRate = (newDistance - oldDistance) / STEP_SIZE
            reward = moveRate
        else:
            reward = -(MAX_NODES - len(nodes))
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
def ReinforcementLearning(epsilon):
    ACNet = ACNetWork()
    ActorOptimizer = torch.optim.Adam(ACNet.ActorModel.parameters(),lr=0.001)
    CriticOptimizer = torch.optim.Adam(ACNet.CriticModel.parameters(),lr=0.01)
    lossFunction = torch.nn.MSELoss()

    mapInfo = GenerateGridMap()
    dataPool = []

    for epoch in range(EPOCHS):
        epsilon = max(epsilon * 0.999, 0.01)
        dataNum = 0
        while dataNum < 1000:
            nodes = [Node(START)]               # 初始化
            state = GetState(nodes, mapInfo)
            nextState = state
            over = False
            while not over:
                action = GetAction(nextState, ACNet)
                state = nextState
                nodes, reward, over = Step(nodes, action, mapInfo)
                nextState = GetState(nodes, mapInfo)
                # state:张量, action:张量, reward:浮点数, nextState:张量, over:布尔值
                dataPool.append((state, action, reward, nextState, over))
                dataNum += 1

        while len(dataPool) > 50000:
            dataPool.pop(0)
    
        # off-policy training
        for _ in range(100):
            Train(dataPool, ACNet, ActorOptimizer, CriticOptimizer, lossFunction)
            print(f'epoch: {epoch}, trainNum: {_}')
        
        # print loss value
        print(f'epoch: {epoch}, ActorLoss: {sum(ACNet.ActorLoss)}, CriticLoss: {sum(ACNet.CriticLoss)}')
        ACNet.initialize()

        # plan on this map
        Play(mapInfo, ACNet)
        
        # plan on new map
        newMapInfo = GenerateGridMap()
        print('new map')
        Play(newMapInfo, ACNet)

# play
def Play(mapInfo, ACNet):
    nodes = [Node(START)]
    sumReward = 0
    for nodeNumber in range(MAX_NODES):
        state = GetState(nodes, mapInfo)
        action = GetAction(state, ACNet)
        nodes, reward, over = Step(nodes, action, mapInfo)
        sumReward += reward
        if over and nodes[-1].point == END:
            print(f'nodeNumber: {nodeNumber}, pathDistance: {calDistance(nodes)}, sumReward: {sumReward}')
            break
        else:
            print('collision')
    if not over:
        print('no path')
    DrawRRT(nodes, mapInfo)

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
    ReinforcementLearning(EPSILON)

if __name__ == '__main__':
    main()