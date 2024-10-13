import Enviroment
import NetWork
import Utils
from torch.utils.tensorboard import SummaryWriter

def modelTrain(path=None):
    PPnets = NetWork.TrainNet("cuda")
    if path is not None:
        PPnets.LoadModel(path)
    
    writer = SummaryWriter('Log')
    global_environment = Enviroment.EnviromentClass()
    global_environment.ResetEnviroment(1,"GlobalPic_a")
    Utils.render(global_environment.map_class, "GlobalPic_a", global_environment.nodes)

    for epoch in range(PPnets.hyper_parameter.train_epoch):
        writer.add_scalar('reward-epoch', PPnets.PlayGame(global_environment, epoch), epoch)
        if (epoch + 1) % 1000 == 0:
            PPnets.SaveModel('SaveModel/' + 'DPG-model-' + str(epoch + 1) + '.pth')
            Utils.render(global_environment.map_class, 'PPfigure_' + str(epoch + 1) , global_environment.nodes)

    writer.close()

def modelTest():
    PPnets = NetWork.TrainNet("cuda")
    environment = Enviroment.EnviromentClass()
    environment.ResetEnviroment(1,"GlobalPic_b")
    modelName = 'DPG-model-100000_1.pth'
    PPnets.LoadModel('SaveModel/' + modelName)

    PPnets.PlayGame(environment, 0)
    Utils.render(environment.map_class, "modelTest", environment.nodes)

# main
def main():
    typeParameter = 0
    if typeParameter == 0:
        modelTrain()
    if typeParameter == 1:
        modelTest()
    return

if __name__ == '__main__':
    main()