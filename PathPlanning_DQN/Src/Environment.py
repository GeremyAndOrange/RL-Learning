import json
import math
import numpy

# 坐标类
class Node:
    def __init__(self, point, parent = None):
        self.point = point
        self.parent = parent

# 环境类
class Environment:
    def __init__(self, map_name: str) -> None:
        self.MapImport(map_name)
        self.ResetEnvironment()
    
    def ResetEnvironment(self):
        self.ConfigImport()
        self.nodes = [self.start_point]

    def MapImport(self, map_name: str):
        self.map_info = []
        map_path = '../Map/' + map_name + '.txt'
        try:
            with open(map_path, 'r') as file:
                for line in file:
                    grids = line.strip().split()
                    row = [int(grid) for grid in grids]
                    self.map_info.append(row)                                           # 栅格地图
        except Exception as error:
            print(f'Error reading file {map_path}: {error}')
            return

    def ConfigImport(self):
        config_path = 'TrainConfig.json'
        try:
            with open(config_path, 'r') as file:
                config = json.load(file)

                environment_config = config.get("EnvironmentConfig", {})
                self.gamma_1 = environment_config.get("Gamma_1",  0.2)
                self.gamma_2 = environment_config.get("Gamma_2",  0.1)
                self.gamma_3 = environment_config.get("Gamma_3", -0.1)
                self.max_steps = environment_config.get("MaxSteps", 500)
                self.lidar_angle = environment_config.get("LidarAngle", 1.8)            # 角度/度
                self.lidar_length = environment_config.get("LidarLength", 5)            # 长度/米
                self.end_point = Node(environment_config.get("EndPoint", [450,450]))
                self.start_point = Node(environment_config.get("StartPoint", [50,50]))  # 坐标值

                map_config = config.get("MapConfig", {})
                self.map_width = map_config.get("MapWidth", 50)                         # 长度/米
                self.map_height = map_config.get("MapHeight", 50)                       # 长度/米
                self.map_resolution = map_config.get("MapResolution", 0.1)
        except Exception as error:
            print(f'Error Import Config file {config_path}: {error}')
            return

    def StateGet(self):
        def SimulateLidar(origin: Node, angle, length):
            samples = int(360/angle)
            coor_x, coor_y = origin.point[0], origin.point[1]                           # 坐标值

            scan_list = []
            for index in range(samples):
                radians = math.radians(angle * index)
                new_x, new_y = coor_x, coor_y

                distance = 0
                while distance < length:
                    new_x += math.cos(radians)                # 坐标值
                    new_y += math.sin(radians)                # 坐标值
                    new_node = Node([int(new_x), int(new_y)])
                    distance = numpy.linalg.norm(numpy.array(new_node.point) - numpy.array(origin.point)) * self.map_resolution

                    if self.CheckOutSide(new_node):
                        break
                    else:
                        if self.CheckCollision(new_node):
                            break
                
                distance = distance if distance < length else length
                scan_list.append(distance)
            return scan_list
        ##################################################

        state_scan = SimulateLidar(self.nodes[-1], self.lidar_angle, self.lidar_length)
        relative_angle = math.atan2((self.end_point.point[1] - self.nodes[-1].point[1]),(self.end_point.point[0] - self.nodes[-1].point[0]))
        state_scan.append(relative_angle)

        return state_scan

    def RewardGet(self):
        old_distance = numpy.linalg.norm(numpy.array(self.nodes[-2].point) - numpy.array(self.end_point.point))
        new_distance = numpy.linalg.norm(numpy.array(self.nodes[-1].point) - numpy.array(self.end_point.point))

        reward_1 = self.gamma_1 * (old_distance - new_distance) * self.map_resolution
        reward_2 = self.gamma_2 * (self.max_steps - len(self.nodes)) + 10 if self.nodes[-1].point == self.end_point.point else 0
        reward_3 = self.gamma_3 * (self.max_steps - len(self.nodes)) - 10 if self.CheckCollision(self.nodes[-1]) or self.CheckOutSide(self.nodes[-1]) else 0
        
        return reward_1 + reward_2 + reward_3

    def Step(self, action: int):
        over, truncated = False, False
        action_dict = {0: [1,0], 1: [1,1], 2: [0,1], 3: [-1,1], 4: [-1,0], 5: [-1,-1], 6: [0,-1], 7: [1,-1]}

        self_node = self.nodes[-1]
        new_x = self_node.point[0] + action_dict[action][0]
        new_y = self_node.point[1] + action_dict[action][1]
        new_node = Node([new_x, new_y], self_node)
        self.nodes.append(new_node)

        state = self.StateGet()
        reward = self.RewardGet()

        if new_node == self.end_point:
            over = True
        
        if self.CheckCollision(new_node) or self.CheckOutSide(new_node):
            truncated = True

        if len(self.nodes) > self.max_steps:
            over = True

        return state, reward, truncated, over

    def CheckCollision(self, node: Node):
        if self.map_info[node.point[0]][node.point[1]] == 1:
            return True
        else:
            return False

    def CheckOutSide(self, node: Node):
        coor_x, coor_y = node.point[0]*self.map_resolution, node.point[1]*self.map_resolution   # 转换为实际位置
        if 0 <= coor_x < self.map_width and 0 <= coor_y < self.map_height:
            return False
        else:
            return True