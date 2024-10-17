import json
import numpy
import random
import matplotlib.pyplot

def Render(environment, pic_name):
    matplotlib.pyplot.figure(figsize=(10, 10))
    final_path = []
    node = environment.nodes[-1]

    while node.parent is not None:
        final_path.append(node.point)
        node = node.parent
    final_path.reverse()

    for item in range(len(final_path) - 1):
        matplotlib.pyplot.plot([final_path[item][0], final_path[item + 1][0]], [final_path[item][1], final_path[item + 1][1]], 'k-')

    for row in range(len(environment.map_info)):
        for column in range(len(environment.map_info[row])):
            if environment.map_info[row][column] == 1:
                # 对每一个障碍物格子填色
                rectangle = matplotlib.pyplot.Rectangle((row, column), environment.map_resolution, environment.map_resolution, edgecolor='blue', facecolor='blue')
                matplotlib.pyplot.gca().add_patch(rectangle)

    matplotlib.pyplot.plot(environment.start_point.point[0], environment.start_point.point[1], 'go')
    matplotlib.pyplot.plot(environment.end_point.point[0], environment.end_point.point[1], 'ro')
    matplotlib.pyplot.xlim(0, int(environment.map_width / environment.map_resolution + 1))
    matplotlib.pyplot.ylim(0, int(environment.map_height / environment.map_resolution + 1))
    matplotlib.pyplot.gca().set_aspect('equal', adjustable='box')

    path = "../Map/" + pic_name + ".png"
    matplotlib.pyplot.savefig(path, dpi=1200)

class GenerateMap:
    def __init__(self) -> None:
        self.ConfigImport()
    
    def ConfigImport(self):
        config_path = 'TrainConfig.json'
        try:
            with open(config_path, 'r') as file:
                config = json.load(file)

                generate_map_config = config.get("GenerateMap", {})
                self.obstacle_num = generate_map_config.get("ObstacleNumber",  12)
                self.obstacle_range = generate_map_config.get("ObstacleRange",  12)

                environment_config = config.get("EnvironmentConfig", {})
                self.end_point = environment_config.get("EndPoint", [450,450])
                self.start_point = environment_config.get("StartPoint", [50,50])        # 坐标值

                map_config = config.get("MapConfig", {})
                self.map_width = map_config.get("MapWidth", 50)                         # 长度/米
                self.map_height = map_config.get("MapHeight", 50)                       # 长度/米
                self.map_resolution = map_config.get("MapResolution", 0.1)
        except Exception as error:
            print(f'Error Import Config file {config_path}: {error}')
            return

    def GenerateObstacle(self):
        def RandomCircle():
            circle_x = random.randint(self.obstacle_range, int(self.map_width/self.map_resolution) - self.obstacle_range)
            circle_y = random.randint(self.obstacle_range, int(self.map_height/self.map_resolution) - self.obstacle_range)
            return [circle_x, circle_y]
        obstacles = []
        for _ in range(self.obstacle_num):
            circle = RandomCircle()
            while numpy.linalg.norm(numpy.array(self.start_point) - numpy.array(circle)) < self.obstacle_range or numpy.linalg.norm(numpy.array(self.end_point) - numpy.array(circle)) < self.obstacle_range:
                circle = RandomCircle()
            obstacles.append(circle)
        return obstacles
    
    def ExportMap(self, map_name):
        obstacles = self.GenerateObstacle()

        mapInfo = numpy.zeros((int(self.map_width/self.map_resolution+1), int(self.map_height/self.map_resolution+1)), dtype=int)
        for obstacle in obstacles:
            for x in range(obstacle[0] - self.obstacle_range, obstacle[0] + self.obstacle_range + 1):
                for y in range(obstacle[1] - self.obstacle_range, obstacle[1] + self.obstacle_range + 1):
                    if numpy.linalg.norm(numpy.array(obstacle) - numpy.array([x, y])) <= self.obstacle_range:
                        mapInfo[x][y] = 1

        map_path = '../Map/' + map_name + ".txt"
        with open(map_path, 'w') as file:
            for row in mapInfo:
                file.write(' '.join(map(str, row)) + '\n')
        return