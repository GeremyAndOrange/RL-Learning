import numpy
import matplotlib.pyplot

def calDistance(nodes) -> float:
    node = nodes[-1]
    distance = 0
    while node.parent is not None:
        distance += numpy.linalg.norm(numpy.array(node.point) - numpy.array(node.parent.point))
        node = node.parent
    return distance

def render(map_class, pic_name, nodes) -> None:
    matplotlib.pyplot.figure(figsize=(10, 10))
    final_path = []
    node = nodes[-1]
    while node.parent is not None:
        final_path.append(node.point)
        node = node.parent
    final_path.reverse()

    for node in nodes:
        if node.parent is not None and node not in final_path:
            matplotlib.pyplot.plot([int(node.point[0] / map_class.resolution), int(node.parent.point[0] / map_class.resolution)], [int(node.point[1] / map_class.resolution), int(node.parent.point[1] / map_class.resolution)], 'r-')
    for item in range(len(final_path) - 1):
        matplotlib.pyplot.plot([int(final_path[item][0] / map_class.resolution), int(final_path[item + 1][0] / map_class.resolution)], [int(final_path[item][1] / map_class.resolution), int(final_path[item + 1][1] / map_class.resolution)], 'k-')

    for row in range(len(map_class.map_info)):
        for column in range(len(map_class.map_info[row])):
            if map_class.map_info[row][column] == 1:
                # 对每一个障碍物格子填色
                rectangle = matplotlib.pyplot.Rectangle((row, column), map_class.resolution, map_class.resolution, edgecolor='blue', facecolor='blue')
                matplotlib.pyplot.gca().add_patch(rectangle)

    matplotlib.pyplot.plot(int(map_class.start_point.point[0] / map_class.resolution), int(map_class.start_point.point[1] / map_class.resolution), 'go')
    matplotlib.pyplot.plot(int(map_class.end_point.point[0] / map_class.resolution), int(map_class.end_point.point[1] / map_class.resolution), 'go')
    matplotlib.pyplot.xlim(0, int(map_class.width / map_class.resolution + 1))
    matplotlib.pyplot.ylim(0, int(map_class.height / map_class.resolution + 1))
    matplotlib.pyplot.gca().set_aspect('equal', adjustable='box')
    
    path = "C:\\Users\\60520\\Desktop\\RL-learning\\PathPlanning\\Map\\" + pic_name
    matplotlib.pyplot.savefig(path, dpi=1200)
    return