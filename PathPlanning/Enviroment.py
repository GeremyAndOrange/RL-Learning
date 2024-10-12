import json
import math
import numpy
import random
import torch

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

    def ConfigExport(self) -> None:
        config_path = 'Config/TrainConfig.json'
        try:
            with open(config_path, 'r') as file:
                config = json.load(file)
        except:
            config = {}

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
        config_path = 'Config/TrainConfig.json'
        try:
            with open(config_path, 'r') as file:
                config = json.load(file)
                map_config = config.get("MapConfig", {})

            self.width = map_config.get("MapWidth", 50)
            self.height = map_config.get("MapHeight", 50)
            self.resolution = map_config.get("MapResolution", 0.5)
            self.obstacle_num = map_config.get("ObstacleNumber", 12)
            self.obstacle_range = map_config.get("ObstacleRange", 4)
            self.start_point = Node(map_config.get("StartPoint", [5,5]))
            self.end_point = Node(map_config.get("EndPoint", [45,45]))
        except:
            return
        return

    def MapExport(self, map_name:str) -> None:
        map_path = 'Map/' + map_name + ".txt"
        with open(map_path, 'w') as file:
            for row in self.map_info:
                file.write(' '.join(map(str, row)) + '\n')
        return

    def MapImport(self, map_name:str) -> None:
        self.map_info = []
        map_path = 'Map/' + map_name + ".txt"
        try:
            with open(map_path, 'r') as file:
                for line in file:
                    numbers = line.strip().split()
                    row = [float(num) for num in numbers]
                    self.map_info.append(row)
        except:
            return
        return

# 环境类
class EnviromentClass:
    def __init__(self) -> None:
        self.map_class = MapClass()
        self.nodes = [self.map_class.start_point]
        self.ConfigImport()

    def ResetEnviroment(self, type:int, map_name:str = None) -> None:
        '''
        :param type:
            0: random generate new map;
            1: import old map;
        :param map_name:
            neccencery if type == 1
        '''
        self.map_class.ConfigImport()
        if type == 0:
            self.map_class.GenerateMap()
        elif type == 1 and map_name is not None:
            self.map_class.MapImport(map_name)
        else :
            return
        self.nodes = [self.map_class.start_point]
        return

    def SimulateLidar(self, grid, origin, angle_step, max_range):
        '''    
        :param grid: map_info。
        :param origin: self_position。
        :param angle_step: sampling degree。
        :param max_range: scan_range。
        '''
        distances = []
        x0, y0 = origin
        num_samples = int(360 / angle_step)
        
        for i in range(num_samples):
            angle = math.radians(angle_step * i)
            distance = 0

            while distance < max_range:
                x = x0 + distance * math.cos(angle)
                y = y0 + distance * math.sin(angle)

                if 0 <= int(x) < len(grid[0]) and 0 <= int(y) < len(grid):
                    if grid[int(y)][int(x)] == COLLISION:
                        break
                else:
                    break
                
                distance += 1
            
            distances.append(distance * self.map_class.resolution)
        
        return distances

    def StateGet(self) -> list:
        self_position = [int(item / self.map_class.resolution) for item in self.nodes[-1].point]
        distance = self.SimulateLidar(self.map_class.map_info, self_position, self.scan_degree, self.scan_range)
        relative_degree = math.atan2((self.map_class.end_point.point[1] - self_position[1]),(self.map_class.end_point.point[0] - self_position[0]))

        state = [(distance, relative_degree)]
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
            reward_2 = self.beta * (self.max_steps - len(self.nodes)) + 10
        if self.map_class.CheckCollision(self.nodes[-1]):
            reward_3 = self.gamma * (self.max_steps - len(self.nodes)) - 10

        reward = reward_1 + reward_2 + reward_3
        return reward

    def ConfigExport(self) -> None:
        config_path = 'Config/TrainConfig.json'
        try:
            with open(config_path, 'r') as file:
                config = json.load(file)
        except:
            config = {}

        config["EnviromentConfig"] = {
            "MaxSteps": self.max_steps,
            "ScanRange": self.scan_range,
            "ScanDegree": self.scan_degree,
            "StepLength": self.step_length,
            "RewardAlpha": self.alpha,
            "RewardBeta": self.beta,
            "RewardGamma": self.gamma
        }

        with open(config_path, 'w') as file:
            json.dump(config, file, indent=4)
        return

    def ConfigImport(self) -> None:
        config_path = 'Config/TrainConfig.json'
        try:
            with open(config_path, 'r') as file:
                config = json.load(file)
                environment_config = config.get("EnviromentConfig", {})
            
            self.max_steps = environment_config.get("MaxSteps", 500)
            self.scan_range = environment_config.get("ScanRange", 50)
            self.scan_degree = environment_config.get("ScanDegree", 7.2)
            self.step_length = environment_config.get("StepLength", 0.5)
            self.alpha = environment_config.get("RewardAlpha", 0.2)
            self.beta = environment_config.get("RewardBeta", 0.1)
            self.gamma = environment_config.get("RewardGamma", -0.1)
        except:
            return
        return