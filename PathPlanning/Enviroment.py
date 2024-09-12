import json
import math
import numpy
import random

COLLISION = 1
OUT_MAP = -1
SPACE = 0

# 坐标类
class Node:
    def __init__(self, point, parent = None):
        self.point = point
        self.parent = parent

# 地图类
class MapClass:
    def __init__(self) -> None:
        self.width = 50
        self.height = 50
        self.resolution = 0.5
        self.obstacle_num = 12
        self.obstacle_range = 4
        self.start_point = Node([5,5])
        self.end_point = Node([45,45])
        self.ConfigImport()
        self.GenerateMap()

    def GenerateMap(self) -> None:
        obstacles = self.GenerateObstacle()

        mapInfo = numpy.zeros((int(self.width / self.resolution + 1), int(self.height / self.resolution + 1)))
        for obstacle in obstacles:
            circleCenter = [int(obstacle[0] / self.resolution), int(obstacle[1] / self.resolution)]
            circleRadius = int(self.obstacle_range / self.resolution)

            for x in range(circleCenter[0] - circleRadius, circleCenter[0] + circleRadius + 1):
                for y in range(circleCenter[1] - circleRadius, circleCenter[1] + circleRadius + 1):
                    if numpy.linalg.norm(numpy.array(circleCenter) - numpy.array([x, y])) <= circleRadius:
                        mapInfo[x][y] = COLLISION
        self.map_info = mapInfo.tolist()
        return

    def GenerateObstacle(self) -> list:
        obstacles = []
        for _ in range(self.obstacle_num):
            circle = [random.randint(self.obstacle_range, self.width - self.obstacle_range), random.randint(self.obstacle_range, self.height - self.obstacle_range)]
            while numpy.linalg.norm(numpy.array(self.start_point.point) - numpy.array(circle)) < self.obstacle_range or numpy.linalg.norm(numpy.array(self.end_point.point) - numpy.array(circle)) < self.obstacle_range:
                circle = [random.randint(self.obstacle_range, self.width - self.obstacle_range), random.randint(self.obstacle_range, self.height - self.obstacle_range)]
            obstacles.append(circle)
        return obstacles

    def CheckCollision(self, node:Node) -> bool:
        point = node.point
        if point[0] < 0 or point[0] > self.width or point[1] < 0 or point[1] > self.height:
            return False
        if self.map_info[int(point[0] / self.resolution)][int(point[1] / self.resolution)] == COLLISION:
            return True
        return False

    def RandomParameter(self) -> None:
        self.width = random.randint(40,60)
        self.height = random.randint(40,60)
        self.resolution = random.choice([0.1, 0.2, 0.5, 1])
        self.obstacle_num = random.randint(10,20)
        self.obstacle_range = random.randint(3,6)
        self.start_point = Node([random.randint(0,self.width), random.randint(0,self.height)])
        self.end_point = Node([random.randint(0,self.width), random.randint(0,self.height)])
        self.ConfigExport()
        return

    def ConfigExport(self) -> None:
        config_path = 'C:\\Users\\60520\\Desktop\\RL-learning\\PathPlanning\\Config\\TrainConfig.json'
        with open(config_path, 'r') as file:
            config = json.load(file)
        
        config["MapConfig"] = {
            "MapWidth": self.width,
            "MapHeight": self.height,
            "MapResolution": self.resolution,
            "ObstacleNumber": self.obstacle_num,
            "ObstacleRange": self.obstacle_range,
            "StartPoint": self.start_point.point,
            "EndPoint": self.end_point.point
        }

        with open(config_path, 'w') as file:
            json.dump(config, file, indent=4)
        return

    def ConfigImport(self) -> None:
        config_path = 'C:\\Users\\60520\\Desktop\\RL-learning\\PathPlanning\\Config\\TrainConfig.json'
        with open(config_path, 'r') as file:
            config = json.load(file)["MapConfig"]
        
        self.width = config["MapWidth"]
        self.height = config["MapHeight"]
        self.resolution = config["MapResolution"]
        self.obstacle_num = config["ObstacleNumber"]
        self.obstacle_range = config["ObstacleRange"]
        self.start_point = Node(config["StartPoint"])
        self.end_point = Node(config["EndPoint"])
        return

    def MapExport(self, map_name:str) -> None:
        map_path = 'C:\\Users\\60520\\Desktop\\RL-learning\\PathPlanning\\Map\\' + map_name
        with open(map_path, 'r') as file:
            for row in self.map_info:
                file.write(' '.join(map(str, row)) + '\n')
        return

    def MapImport(self, map_name:str) -> None:
        map_path = 'C:\\Users\\60520\\Desktop\\RL-learning\\PathPlanning\\Map\\' + map_name
        with open(map_path, 'w') as file:
            for line in file:
                numbers = line.strip().split()
                row = [int(num) for num in numbers]
                self.map_info.append(row)
        return

# 环境类
class EnviromentClass:
    def __init__(self) -> None:
        self.map_class = MapClass()
        self.nodes = [self.map_class.start_point]
        self.max_steps = 500
        self.scan_range = 50
        self.step_length = 0.5
        self.alpha = 0.1
        self.beta = 0.1
        self.gamma = 0.1
        self.ConfigImport()

    def ResetEnviroment(self) -> None:
        self.map_class.ConfigImport()
        self.map_class.GenerateMap()
        self.nodes = [self.map_class.start_point]
        return

    def StateGet(self) -> tuple:
        self_position = [int(item / self.map_class.resolution) for item in self.nodes[-1].point]
        state_map = [[SPACE for _ in range(2 * self.scan_range + 1)] for _ in range(2 * self.scan_range + 1)]

        for x in range(2 * self.scan_range + 1):
            for y in range(2 * self.scan_range + 1):
                map_x = x + (self_position[0] - self.scan_range)
                map_y = y + (self_position[1] - self.scan_range)
                if map_x >= 0 and map_x < len(self.map_class.map_info[0]) and map_y >= 0 and map_y < len(self.map_class.map_info):
                    state_map[x][y] = self.map_class.map_info[map_x][map_y]
                else:
                    state_map[x][y] = OUT_MAP

        state = [(state_map, self.nodes[-1].point, self.map_class.end_point.point)]
        return state

    def Step(self, action:float):
        over, truncated = False, False
        if numpy.linalg.norm(numpy.array(self.nodes[-1].point) - numpy.array(self.map_class.end_point.point)) < self.step_length:
            new_node = Node(list(self.map_class.end_point.point), self.nodes[-1])
            self.nodes.append(new_node)
            over, truncated = True, True
        else:
            new_point = numpy.array(self.nodes[-1].point) + numpy.array([self.step_length * math.cos(action / 180 * math.pi), self.step_length * math.sin(action / 180 * math.pi)])
            new_node = Node(list(new_point), self.nodes[-1])
            self.nodes.append(new_node)
            if new_point[0] >= 0 and new_point[0] <= self.map_class.width and new_point[1] >= 0 and new_point[1] <= self.map_class.height:
                if self.map_class.CheckCollision(new_node):
                    truncated = True
            else:
                truncated = True
        if len(self.nodes) > self.max_steps:
            over = True

        reward = self.RewardGet()
        state = self.StateGet()
        return state, reward, truncated, over

    def RewardGet(self) -> float:
        reward_1, reward_2, reward_3 = 0, 0, 0

        old_distance = numpy.linalg.norm(numpy.array(self.nodes[-2].point) - numpy.array(self.map_class.end_point.point))
        new_distance = numpy.linalg.norm(numpy.array(self.nodes[-1].point) - numpy.array(self.map_class.end_point.point))
        reward_1 = self.alpha * (old_distance - new_distance)
        if self.nodes[-1].point == self.map_class.end_point.point:
            reward_2 = self.beta * (self.max_steps - len(self.nodes))
        if self.map_class.CheckCollision(self.nodes[-1]):
            reward_3 = -self.beta * (self.max_steps - len(self.nodes)) * self.gamma

        reward = reward_1 + reward_2 + reward_3
        return reward

    def ConfigExport(self) -> None:
        config_path = 'C:\\Users\\60520\\Desktop\\RL-learning\\PathPlanning\\Config\\TrainConfig.json'
        with open(config_path, 'r') as file:
            config = json.load(file)

        config["EnviromentConfig"] = {
            "MaxSteps": self.max_steps,
            "ScanRange": self.scan_range,
            "StepLength": self.step_length,
            "RewardAlpha": self.alpha,
            "RewardBeta": self.beta,
            "RewardGamma": self.gamma
        }

        with open(config_path, 'w') as file:
            json.dump(config, file, indent=4)
        return

    def ConfigImport(self) -> None:
        config_path = 'C:\\Users\\60520\\Desktop\\RL-learning\\PathPlanning\\Config\\TrainConfig.json'
        with open(config_path, 'r') as file:
            config = json.load(file)["EnviromentConfig"]
        
        self.max_steps = config["MaxSteps"]
        self.scan_range = config["ScanRange"]
        self.step_length = config["StepLength"]
        self.alpha = config["RewardAlpha"]
        self.beta = config["RewardBeta"]
        self.gamma = config["RewardGamma"]
        return