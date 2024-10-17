import Utils
import NetWork
import Environment
from torch.utils.tensorboard import SummaryWriter

def ModelTrain(path=None):
    PPnets = NetWork.TrainNetWork('cuda')
    if path is not None:
        PPnets.LoadModel(path)
    
    # writer = SummaryWriter('Log')
    global_environment = Environment.Environment('GlobalPic_a')
    Utils.Render(global_environment, "GlobalPic_a")

    while len(PPnets.data_store) < PPnets.data_select:
        PPnets.PlayGame(global_environment)
    
    PPnets.Initialize()
    for epoch in range(PPnets.train_epoch):
        PPnets.TrainNet()
        sum_reward = PPnets.PlayGame(global_environment, epoch)
        PPnets.EpsilonFunction()
        PPnets.Initialize()
        # writer.add_scalar('reward-epoch', sum_reward, epoch)
        if epoch % 10 == 0:
            PPnets.UpdateTargetNetWork(PPnets.target_net, PPnets.advantage_net)
        if (epoch + 1) % 10 == 0:
            PPnets.SaveModel('../Model/' + 'model-' + str(epoch + 1) + '.pth')
            Utils.Render(global_environment, 'PPfigure_' + str(epoch + 1))
    
    # writer.close()

def ModelTest():
    PPnets = NetWork.TrainNetWork('cuda')
    environment = Environment.Environment('GlobalPic_b')
    modelName = 'Dmodel-100000.pth'
    PPnets.LoadModel('../Model/' + modelName)

    PPnets.PlayGame(environment)
    Utils.Render(environment, "modelTest")

# main
def main():
    typeParameter = 0
    if typeParameter == 0:
        ModelTrain()
    if typeParameter == 1:
        ModelTest()
    return

if __name__ == '__main__':
    main()