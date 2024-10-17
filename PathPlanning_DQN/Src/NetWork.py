import json
import numpy
import torch
import random

# DoubleDQN
class DDQN(torch.nn.Module):
    def __init__(self, device, in_dim: int) -> None:
        super(DDQN,self).__init__()
        self.advantageNetWork = torch.nn.Sequential(
            torch.nn.Linear(in_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 8)
        ).to(device)

        self.valueNetWork = torch.nn.Sequential(
            torch.nn.Linear(in_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1)
        ).to(device)
        self.device = device
    
    def forward(self, state):
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        # print(state.shape)
        advantage = self.advantageNetWork(state)
        value = self.valueNetWork(state)
        QValue = value + advantage - advantage.mean()
        return QValue

# 训练网络类
class TrainNetWork:
    def __init__(self, device) -> None:
        self.device = device
        self.data_store = []
        self.sum_reward_list = []
        self.ImportConfig()                                                 # 导入超参数

        self.advantage_net = DDQN(device, int(360/self.lidar_angle) + 1)
        self.target_net = DDQN(device, int(360/self.lidar_angle) + 1)

        self.Initialize()                                                   # 初始化记录列表
        self.UpdateTargetNetWork(self.target_net, self.advantage_net)       # 保持网络参数一致
        self.lossFunction = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.advantage_net.parameters(), lr=self.lr)

    def Initialize(self):
        self.reward = []
        self.loss = []

    def UpdateTargetNetWork(self, TargetNetWork, NetWork):
        TargetNetWork.load_state_dict(NetWork.state_dict())
    
    def ImportConfig(self):
        config_path = 'TrainConfig.json'
        try:
            with open(config_path, 'r') as file:
                config = json.load(file)

                hyperparameter_config = config.get("HyperParameters", {})
                self.lr = hyperparameter_config.get("LearningRate",  0.001)
                self.epsilon = hyperparameter_config.get("Epsilon",  0.999)
                self.data_max = hyperparameter_config.get("DataMax", 1000000)
                self.data_select = hyperparameter_config.get("DataSelect", 256)
                self.train_epoch = hyperparameter_config.get("TrainEpoch", 10000)
                self.discount_rate = hyperparameter_config.get("DiscountRate", 0.99)

                environment_config = config.get("EnvironmentConfig", {})
                self.lidar_angle = environment_config.get("LidarAngle", 1.8)            # 角度/度
        except Exception as error:
            print(f'Error Import Config file {config_path}: {error}')
            return

    def SaveModel(self, path):
        torch.save({
            'Target_Net': self.target_net.state_dict(),
            'Advantage_Net': self.advantage_net.state_dict(),
        }, path)
    
    def LoadModel(self, path):
        checkpoint = torch.load(path, weights_only=True)
        self.target_net.load_state_dict(checkpoint['Target_Net'])
        self.advantage_net.load_state_dict(checkpoint['Advantage_Net'])

    def EpsilonFunction(self):
        self.sum_reward_list.append(sum(self.reward))

        self.epsilon = max(self.epsilon * 0.9995, 0.05)

    def ActionForward(self, state):
        if random.random() < self.epsilon:
            action = random.randint(0, 7)
        else:
            advantage_list = self.advantage_net(state)
            action = torch.argmax(advantage_list).item()
        return action
    
    def TrainNet(self):
        sample_data = random.sample(self.data_store, self.data_select)
        state = numpy.array([data[0] for data in sample_data])
        action = torch.tensor(numpy.array([data[1] for data in sample_data]), dtype=torch.int64).reshape(-1,1).to(self.device)
        reward = torch.tensor(numpy.array([data[2] for data in sample_data]), dtype=torch.float32).reshape(-1,1).to(self.device)
        next_state = numpy.array([data[3] for data in sample_data])
        over = torch.tensor(numpy.array([data[4] for data in sample_data]), dtype=torch.int64).reshape(-1,1).to(self.device)

        # print(action.shape, reward.shape, over.shape)
        QValue = self.advantage_net(state).gather(1, action)
        with torch.no_grad():
            next_action = self.advantage_net(next_state).argmax(dim=1).reshape(-1,1)
            next_QValue = self.target_net(next_state).gather(1, next_action)
            target = reward + self.discount_rate * next_QValue * (1 - over)
        loss = self.lossFunction(QValue, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.loss.append(loss.item())

    def PlayGame(self, environment, epoch=None):
        over = False
        environment.ResetEnvironment()
        state = environment.StateGet()
        while not over:
            action = self.ActionForward(state)
            next_sate, reward, truncated, finished = environment.Step(action)
            over = finished or truncated
            self.data_store.append((state, action, reward, next_sate, over))

            state = next_sate
            self.reward.append(reward)

        while len(self.data_store) > self.data_max:
            self.data_store.pop(0)
        sum_reward = sum(self.reward)
        if epoch is not None:
            print(f'Epoch: {epoch}, Loss: {sum(self.loss)}, Reward: {sum_reward}')
        return sum_reward