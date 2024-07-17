import numpy
import random
import matplotlib.pyplot

class Node:
    def __init__(self, point, parent = None):
        self.point = point
        self.parent = parent

# parameters
START = [0, 0]          # 起点坐标
END = [50, 50]          # 终点坐标
MAX_NODES = 5000        # 节点最大数量
STEP_SIZE = 0.5         # 步长
OBSTACLE_NUM = 5        # 障碍物数量
OBSTACLE_RADIUS = 5     # 障碍物半径

# general obstacles
def GeneralObstacles():
    obstacles = []
    for _ in range(OBSTACLE_NUM):
        circle = [random.randint(0, 50), random.randint(0, 50)]
        while numpy.linalg.norm(numpy.array(START) - numpy.array(circle)) < OBSTACLE_RADIUS or numpy.linalg.norm(numpy.array(END) - numpy.array(circle)) < OBSTACLE_RADIUS:
            circle = [random.randint(0, 50), random.randint(0, 50)]
        obstacles.append(circle)
    return obstacles

# check collision
def CheckCollision(node, obstacles):
    point = numpy.array(node.point)
    for obstacle in obstacles:
        if numpy.linalg.norm(point - numpy.array(obstacle)) < OBSTACLE_RADIUS:
            return True
    return False

# generate random node
def GenerateRandomNode(obstacles):
    if random.random() < 0.5:
        while True:
            point = [random.randint(0, 50), random.randint(0, 50)]
            node = Node(point)
            if not CheckCollision(node, obstacles):
                return node
    else:
        return Node(END)

# find nearest node
def FindNearest(nodes, node):
    point = numpy.array(node.point)
    minDistance = float('inf')
    nearestNode = None
    for node in nodes:
        distance = numpy.linalg.norm(point - numpy.array(node.point))
        if distance < minDistance:
            minDistance = distance
            nearestNode = node
    return nearestNode

# generate rrt
def GenerateRRT(obstacles):
    nodes = [Node(START)]
    for _ in range(MAX_NODES):
        randomNode = GenerateRandomNode(obstacles)
        nearestNode = FindNearest(nodes, randomNode)
        # extend
        if numpy.linalg.norm(numpy.array(randomNode.point) - numpy.array(nearestNode.point)) < STEP_SIZE:
            newNode = Node(list(randomNode.point), nearestNode)
            if not CheckCollision(newNode, obstacles):
                nodes.append(newNode)
        else:
            newPoint = nearestNode.point + (numpy.array(randomNode.point) - numpy.array(nearestNode.point)) / numpy.linalg.norm(numpy.array(randomNode.point) - numpy.array(nearestNode.point)) * STEP_SIZE
            newNode = Node(list(newPoint), nearestNode)
            if not CheckCollision(newNode, obstacles):
                nodes.append(newNode)
    return nodes

def DrawRRT(nodes, obstacles):
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

    for obstacle in obstacles:
        circle = matplotlib.pyplot.Circle(obstacle, OBSTACLE_RADIUS, color='b', fill=True)
        matplotlib.pyplot.gca().add_patch(circle)
    
    matplotlib.pyplot.plot(START[0], START[1], 'go')
    matplotlib.pyplot.plot(END[0], END[1], 'go')
    matplotlib.pyplot.xlim(0, 50)
    matplotlib.pyplot.ylim(0, 50)
    matplotlib.pyplot.gca().set_aspect('equal', adjustable='box')
    matplotlib.pyplot.show()

def calDistance(nodes):
    node = nodes[-1]
    distance = 0
    while node.parent is not None:
        distance += numpy.linalg.norm(numpy.array(node.point) - numpy.array(node.parent.point))
        node = node.parent
    return distance

obstacles = GeneralObstacles()
nodes = GenerateRRT(obstacles)
print(calDistance(nodes))
DrawRRT(nodes, obstacles)