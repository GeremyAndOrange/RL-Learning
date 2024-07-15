import math
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
MAX_NODES = 200         # 节点最大数量
STEP_SIZE = 0.5         # 步长
OBSTACLE_NUM = 5        # 障碍物数量
OBSTACLE_RADIUS = 5     # 障碍物半径
MAP_LENGTH = 50         # 地图长度
MAP_HEIGHT = 50         # 地图宽度
MAP_RESOLUTION = 0.5    # 地图分辨率
EPOCHS = 1000           # 训练轮数
EPSILON = 0.99          # epsilon-greedy
SCAN_RANGE = 99         # 智能体扫描范围
ALPHA = 3               # 奖励权重

# enumeration value
COLLISION = 255
ROBOT = 127

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
    if mapInfo[int(point[0] / MAP_RESOLUTION)][int(point[1] / MAP_RESOLUTION)] == COLLISION:
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
                    mapInfo[x][y] = COLLISION
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
            torch.nn.Linear((SCAN_RANGE+1)**2, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 4),
            torch.nn.ReLU(),
            torch.nn.Linear(4, 1),
            torch.nn.Tanh()
        ]

        CriticStateLayer = [
            torch.nn.Linear((SCAN_RANGE+1)**2, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 63)
        ]
        CriticActionLayer = [
            torch.nn.Linear(64, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 4),
            torch.nn.ReLU(),
            torch.nn.Linear(4, 1)
        ]

        self.ActorModel = torch.nn.Sequential(*ActorLayer)
        self.CriticStateModel = torch.nn.Sequential(*CriticStateLayer)
        self.CriticActionModel = torch.nn.Sequential(*CriticActionLayer)
        self.device = device
        self.initialize()

    def ActorForward(self, state):
        state = state.view(1,(SCAN_RANGE+1)**2)
        return self.ActorModel(state) * 360
    
    def CriticForward(self, state, action):
        state = state.view(1,(SCAN_RANGE+1)**2)
        stateFeature = self.CriticStateModel(state)
        tensor = torch.cat((stateFeature.squeeze(0), action), 0)
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
    # train
    Qvalue = ACNet.CriticForward(state, action)
    with torch.no_grad():
        nextAction = ACNet.ActorForward(nextState).squeeze(0)
        nextQvalue = ACNet.CriticForward(nextState, nextAction)
        target = reward + 0.9 * nextQvalue * (1 - over)
    CriticLoss = lossFunction(Qvalue, target)
    ACNet.CriticLoss.append(CriticLoss)
    # 更新评论家网络
    CriticOptimizer.zero_grad()
    CriticLoss.backward()
    CriticOptimizer.step()

    # 用更新后的评论家网络计算演员网络的损失
    Qvalues = ACNet.CriticForward(state, ACNet.ActorForward(state).squeeze(0))
    ActorLoss = -Qvalues.mean()
    ACNet.ActorLoss.append(ActorLoss)
    # 更新演员网络
    ActorOptimizer.zero_grad()
    ActorLoss.backward()
    ActorOptimizer.step()

# get state
def GetState(nodes, mapInfo, ACNet):
    positionInfo = [int(item / MAP_RESOLUTION) for item in nodes[-1].point]
    state = [[0 for _ in range(SCAN_RANGE + 1)] for _ in range(SCAN_RANGE + 1)]

    for x in range(SCAN_RANGE + 1):
        for y in range(SCAN_RANGE + 1):
            map_x = x+(positionInfo[0] - SCAN_RANGE//2)
            map_y = y+(positionInfo[1] - SCAN_RANGE//2)
            if map_x >= 0 and map_x <= int(MAP_LENGTH / MAP_RESOLUTION) and map_y >= 0 and map_y <= int(MAP_HEIGHT / MAP_RESOLUTION) :
                state[x][y] = mapInfo[map_x][map_y]
            else:
                state[x][y] = -1
    state[SCAN_RANGE//2][SCAN_RANGE//2] = ROBOT
    state = torch.tensor(numpy.array(state), dtype=torch.float32).to(ACNet.device)
    # state是一个二维张量
    return state

# get action
def GetAction(state, ACNet, epsilon):
    if random.random() < epsilon:
        action = torch.tensor(random.random()*360, dtype=torch.float32).unsqueeze(0).to(ACNet.device)
    else:
        action = ACNet.ActorForward(state).squeeze(0)
    # action是一个一维张量,长度为1
    return action

# get reward
def GetReward(nodes, mapInfo):
    if nodes[-1].point == END:
        reward = (MAX_NODES - len(nodes)) * 1
    elif (nodes[-1].point[0] >= 0 and nodes[-1].point[0] <= 50 and nodes[-1].point[1] >= 0 and nodes[-1].point[1] <= 50) and CheckCollision(nodes[-1], mapInfo):
        reward = (MAX_NODES - len(nodes)) * (-1)
    else:
        oldDistance = numpy.linalg.norm(numpy.array(nodes[-2].point) - numpy.array(END))
        newDistance = numpy.linalg.norm(numpy.array(nodes[-1].point) - numpy.array(END))
        moveRate = (oldDistance - newDistance) / STEP_SIZE
        reward = ALPHA * moveRate + 0.5

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
        newNode = Node(list(newPoint), nodes[-1])
        nodes.append(newNode)
        if newPoint[0] >= 0 and newPoint[0] <= MAP_LENGTH and newPoint[1] >= 0 and newPoint[1] <= MAP_HEIGHT:
            if CheckCollision(newNode, mapInfo):
                over = True
        else:
            over = True
    reward = GetReward(nodes, mapInfo)

    if len(nodes) > MAX_NODES:
        over = True
    return nodes, reward, over

# reinforcement learning
def ReinforcementLearning(device, epsilon):
    ACNet = ACNetWork(device)
    ACNet.to(device)
    ActorOptimizer = torch.optim.Adam(ACNet.ActorModel.parameters(),lr=0.005)
    CriticOptimizer = torch.optim.Adam(list(ACNet.CriticStateModel.parameters()) + list(ACNet.CriticActionModel.parameters()),lr=0.01)
    lossFunction = torch.nn.MSELoss()

    # mapInfo = GenerateGridMap()
    # numpy.savetxt('map.txt', mapInfo)
    mapInfo = numpy.loadtxt('map.txt')
    
    for epoch in range(EPOCHS):
        # on-policy training
        for _ in range(100):
            epsilon = max(0.05, epsilon * 0.999)
            nodes = [Node(START)]
            state = GetState(nodes, mapInfo, ACNet)
            over = False
            sumReward = 0
            while not over:
                action = GetAction(state, ACNet, epsilon)
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

# play
def Play(mapInfo, ACNet):
    nodes = [Node(START)]
    sumReward = 0
    for nodeNumber in range(MAX_NODES):
        state = GetState(nodes, mapInfo, ACNet)
        action = GetAction(state, ACNet, 0)
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
    
    DrawPath(nodes, mapInfo)

# save train text
def saveTrainText(str):
    with open('train.txt', 'a') as f:
        f.write(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + '\n')
        f.write(str + '\n')
    print(str)

# draw
def DrawPath(nodes, mapInfo):
    matplotlib.pyplot.figure(figsize=(10, 10))
    finalPath = []
    node = nodes[-1]
    while node.parent is not None:
        finalPath.append(node.point)
        node = node.parent
    finalPath.reverse()

    for node in nodes:
        if node.parent is not None and node not in finalPath:
            matplotlib.pyplot.plot([int(node.point[0] / MAP_RESOLUTION), int(node.parent.point[0] / MAP_RESOLUTION)], [int(node.point[1] / MAP_RESOLUTION), int(node.parent.point[1] / MAP_RESOLUTION)], 'r-')

    for item in range(len(finalPath) - 1):
        matplotlib.pyplot.plot([int(finalPath[item][0] / MAP_RESOLUTION), int(finalPath[item + 1][0] / MAP_RESOLUTION)], [int(finalPath[item][1] / MAP_RESOLUTION), int(finalPath[item + 1][1] / MAP_RESOLUTION)], 'k-')

    for row in range(len(mapInfo)):
        for column in range(len(mapInfo[row])):
            if mapInfo[row][column] == COLLISION:
                # 对每一个障碍物格子填色
                rectangle = matplotlib.pyplot.Rectangle((row, column), MAP_RESOLUTION, MAP_RESOLUTION, edgecolor='blue', facecolor='blue')
                matplotlib.pyplot.gca().add_patch(rectangle)
        
    matplotlib.pyplot.plot(int(START[0] / MAP_RESOLUTION), int(START[1] / MAP_RESOLUTION), 'go')
    matplotlib.pyplot.plot(int(END[0] / MAP_RESOLUTION), int(END[1] / MAP_RESOLUTION), 'go')
    matplotlib.pyplot.xlim(0, int(MAP_LENGTH / MAP_RESOLUTION + 1))
    matplotlib.pyplot.ylim(0, int(MAP_HEIGHT / MAP_RESOLUTION + 1))
    matplotlib.pyplot.gca().set_aspect('equal', adjustable='box')
    matplotlib.pyplot.show()

# main
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ReinforcementLearning(device, EPSILON)
    # DrawPath([Node(START)],numpy.loadtxt('map.txt'))

if __name__ == '__main__':
    main()