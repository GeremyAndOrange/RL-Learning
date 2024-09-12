import Enviroment
import NetWork
import Utils
import copy
import time
import threading
from torch.utils.tensorboard import SummaryWriter

def modelTrain(path=None):
    PPnets = NetWork.Central("cuda")
    if path is not None:
        PPnets.main_net.LoadModel(path)
    
    writer = SummaryWriter('C:\\Users\\60520\\Desktop\\RL-learning\\PathPlanning\\Log')
    global_environment = Enviroment.EnviromentClass()
    Utils.render(global_environment.map_class, "GlobalPic.png", global_environment.nodes)

    threads = []
    StartEvent = [threading.Event() for _ in range(PPnets.main_net.hyper_parameter.worker_num)]
    JoinEvents = [threading.Event() for _ in range(PPnets.main_net.hyper_parameter.worker_num)]
    StopEvents = threading.Event()

    for _ in range(PPnets.main_net.hyper_parameter.worker_num):
        thread = threading.Thread(target=workerThread, args=(PPnets, StartEvent[_], JoinEvents[_], StopEvents))
        threads.append(thread)
        thread.start()

    for epoch in range(PPnets.main_net.hyper_parameter.train_epoch):
        for StartEvent_ in StartEvent:
            StartEvent_.set()
        for JoinEvent in JoinEvents:
            JoinEvent.wait()
            JoinEvent.clear()

        PPnets.main_net.TrainNet()
        writer.add_scalar('reward-epoch', PPnets.main_net.PlayGame(global_environment, epoch, 1), epoch)
        if (epoch + 1) % 5000 == 0:
            PPnets.main_net.SaveModel('C:\\Users\\60520\\Desktop\\RL-learning\\PathPlanning\\SaveModel\\' + 'DPG-model-' + str(epoch + 1) + '.pth')
            Utils.render(global_environment.map_class, 'PPfigure_' + str(epoch + 1) + '.png', global_environment.nodes)
    StopEvents.set()
    for thread in threads:
        thread.join()
    writer.close()

def modelTest():
    PPnets = NetWork.TrainNet("cuda")
    environment = Enviroment.EnviromentClass()
    modelName = 'DPG-model-100000_1.pth'
    PPnets.LoadModel('C:\\Users\\60520\\Desktop\\RL-learning\\PathPlanning\\SaveModel\\' + modelName)

    PPnets.PlayGame(environment, 0, 1)
    Utils.render(environment.map_class, "modelTest")

def workerThread(PPnets, StartEvent, JoinEvent, StopEvent, localEnvironment=None):
    worker = NetWork.Worker(copy.deepcopy(PPnets.main_net))
    while not StopEvent.is_set():
        if StartEvent.is_set():
            if localEnvironment == None:
                localEnvironment = Enviroment.EnviromentClass()
            worker.ResetNetState()
            worker.UpdateNetPatameter(PPnets.main_net)
            worker.GenerateData(localEnvironment)
            PPnets.GetData(worker.CommitData())
            JoinEvent.set()
            StartEvent.clear()
        else: 
            time.sleep(0.1)

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