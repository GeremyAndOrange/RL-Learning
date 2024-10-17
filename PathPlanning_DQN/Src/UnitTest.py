import Environment
import NetWork
import Utils

# Generate Map
ultils_map = Utils.GenerateMap()
ultils_map.ExportMap('GlobalPic_b')

# Render
global_environment = Environment.Environment('GlobalPic_b')
Utils.Render(global_environment, 'GlobalPic_b')