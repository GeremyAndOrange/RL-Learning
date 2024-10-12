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
    
    writer = SummaryWriter('Log')
    global_environment = Enviroment.EnviromentClass()
    global_environment.ResetEnviroment(1,"GlobalPic_a")
    Utils.render(global_environment.map_class, "GlobalPic_a", global_environment.nodes)

    threads = []
    StartEvent = [threading.Event() for _ in range(PPnets.main_net.hyper_parameter.worker_num)]
    JoinEvents = [threading.Event() for _ in range(PPnets.main_net.hyper_parameter.worker_num)]
    StopEvents = threading.Event()

    for _ in range(PPnets.main_net.hyper_parameter.worker_num):
        thread = threading.Thread(target=workerThread, args=(PPnets, StartEvent[_], JoinEvents[_], StopEvents))
        threads.append(thread)
        thread.start()

    for epoch in range(PPnets.main_net.hyper_parameter.train_epoch):
        while PPnets.main_net.data_store.Length() < PPnets.main_net.hyper_parameter.data_max:
            for StartEvent_ in StartEvent:
                StartEvent_.set()
            for JoinEvent in JoinEvents:
                JoinEvent.wait()
                JoinEvent.clear()

        PPnets.main_net.TrainNet()
        writer.add_scalar('reward-epoch', PPnets.main_net.PlayGame(global_environment, epoch, 1), epoch)
        if (epoch + 1) % 1000 == 0:
            PPnets.main_net.SaveModel('SaveModel/' + 'DPG-model-' + str(epoch + 1) + '.pth')
            Utils.render(global_environment.map_class, 'PPfigure_' + str(epoch + 1) , global_environment.nodes)
    StopEvents.set()
    for thread in threads:
        thread.join()
    writer.close()

def modelTest():
    PPnets = NetWork.TrainNet("cuda")
    environment = Enviroment.EnviromentClass()
    environment.ResetEnviroment(1,"GlobalPic_b")
    modelName = 'DPG-model-100000_1.pth'
    PPnets.LoadModel('SaveModel/' + modelName)

    PPnets.PlayGame(environment, 0, 1)
    Utils.render(environment.map_class, "modelTest")

def workerThread(PPnets, StartEvent, JoinEvent, StopEvent, localEnvironment=None):
    worker = NetWork.Worker(copy.deepcopy(PPnets.main_net))
    while not StopEvent.is_set():
        if StartEvent.is_set():
            if localEnvironment == None:
                localEnvironment = Enviroment.EnviromentClass()
                localEnvironment.ResetEnviroment(1,"GlobalPic_a")
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