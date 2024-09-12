import json
import numpy
import torch
import random

# 超参数类
class HyperParameters:
    def __init__(self):
        self.actor_lr = 3e-4
        self.critic_lr = 4e-4
        self.gamma = 0.99
        self.data_select = 64
        self.update_num = 10
        self.clamp_value = 0.2
        self.clip_value = 0.5
        self.data_max = 100000
        self.worker_num = 10
        self.train_epoch = 200000
        self.ConfigImport()
        self.GetScanRange()



    def GetScanRange(self):
        config_path = 'C:\\Users\\60520\\Desktop\\RL-learning\\PathPlanning\\Config\\TrainConfig.json'
        with open(config_path, 'r') as file:
            config = json.load(file)["EnviromentConfig"]
        
        self.scan_range = config["ScanRange"]

    def ConfigExport(self) -> None:
        config_path = 'C:\\Users\\60520\\Desktop\\RL-learning\\PathPlanning\\Config\\TrainConfig.json'
        with open(config_path, 'r') as file:
            config = json.load(file)
        
        config["HyperParameters"] = {
            "ActorLearningRate": self.actor_lr,
            "CriticLearningRate": self.critic_lr,
            "DiscountRate": self.gamma,
            "DataSelect": self.data_select,
            "UpdateNumber": self.update_num,
            "ClampValue": self.clamp_value,
            "ClipValue": self.clip_value,
            "DataMax": self.data_max,
            "WorkerNumber": self.worker_num,
            "ScanRange": self.scan_range,
            "TrainEpoch": self.train_epoch
        }

        with open(config_path, 'w') as file:
            json.dump(config, file, indent=4)
        return

    def ConfigImport(self) -> None:
        config_path = 'C:\\Users\\60520\\Desktop\\RL-learning\\PathPlanning\\Config\\TrainConfig.json'
        with open(config_path, 'r') as file:
            config = json.load(file)["HyperParameters"]
        
        self.actor_lr = config["ActorLearningRate"]
        self.critic_lr = config["CriticLearningRate"]
        self.gamma = config["DiscountRate"]
        self.data_select = config["DataSelect"]
        self.update_num = config["UpdateNumber"]
        self.clamp_value = config["ClampValue"]
        self.clip_value = config["ClipValue"]
        self.data_max = config["DataMax"]
        self.worker_num = config["WorkerNumber"]
        self.scan_range = config["ScanRange"]
        self.train_epoch = config["TrainEpoch"]
        return

# 状态特征类
class StateFeature(torch.nn.Module):
    def __init__(self, device:str, width:int, height:int) -> None:
        '''
        device: "cuda", "cpu" \n
        widht:  scan_enviroment_width \n
        height: scan_enviroment_height \n
        '''
        super(StateFeature,self).__init__()
        self.img_netWork = torch.nn.Sequential(
            torch.nn.Linear(width * height, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 2)
        ).to(device)
        self.device = device

    def forward(self, batch_states:list):
        state_maps = torch.tensor(numpy.array([state[0] for state in batch_states]), dtype=torch.float32).to(self.device)
        state_selves = torch.tensor(numpy.array([state[1] for state in batch_states]), dtype=torch.float32).to(self.device)
        state_ends = torch.tensor(numpy.array([state[2] for state in batch_states]), dtype=torch.float32).to(self.device)
        img_states = state_maps.reshape(state_maps.size(0), -1)
        img_features = self.img_netWork(img_states)
        env_features = torch.cat((img_features, state_selves, state_ends), dim=1)
        return env_features

# Actor类
class Actor(torch.nn.Module):
    def __init__(self, device:str) -> None:
        super(Actor,self).__init__()
        self.sigma_netWork = torch.nn.Sequential(
            torch.nn.Linear(6, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1),
            torch.nn.Softplus()
        ).to(device)

        self.mu_netWork = torch.nn.Sequential(
            torch.nn.Linear(6, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1),
            torch.nn.Tanh()
        ).to(device)

    def forward(self, env_feature):
        Mu = self.mu_netWork(env_feature) * 180 + 180
        Sigma = self.sigma_netWork(env_feature)
        return Mu, Sigma

# Critic类
class Critic(torch.nn.Module):
    def __init__(self, device:str) -> None:
        super(Critic,self).__init__()
        self.value_netWork = torch.nn.Sequential(
            torch.nn.Linear(6, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1)
        ).to(device)

    def forward(self, env_feature):
        value = self.value_netWork(env_feature)
        return value

# 仓库类
class Store:
    def __init__(self) -> None:
        self.data_store = []

    def SaveData(self, data:tuple):
        self.data_store.append(data)

    def LoadData(self, number:int) -> list:
        return random.sample(self.data_store, number)

    def ClearData(self):
        self.data_store.clear()

    def Length(self) -> int:
        return len(self.data_store)
    
    def PopData(self):
        self.data_store.pop(0)

# 训练网络类
class TrainNet:
    def __init__(self, device) -> None:
        self.device = device
        self.hyper_parameter = HyperParameters()
        self.env_net = StateFeature(device, 2 * self.hyper_parameter.scan_range + 1, 2 * self.hyper_parameter.scan_range + 1)
        self.actor = Actor(device)
        self.critic = Critic(device)
        self.data_store = Store()

        self.lossFunction = torch.nn.MSELoss()
        self.actor_optimizer = torch.optim.Adam(list(self.actor.parameters()) + list(self.env_net.parameters()), lr=self.hyper_parameter.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.hyper_parameter.critic_lr)

        self.Initialize()

    def Initialize(self):
        self.reward = []
        self.actor_loss = []
        self.critic_loss = []

    def SaveModel(self, path):
        torch.save({
            'ENV_NET_Model_state_dict': self.env_net.state_dict(),
            'ACTOR_Model_state_dict': self.actor.state_dict(),
            'CRITIC_Model_state_dict': self.critic.state_dict(),
        }, path)
        self.hyper_parameter.ConfigExport()
    
    def LoadModel(self, path):
        checkpoint = torch.load(path, weights_only=True)
        self.env_net.load_state_dict(checkpoint['ENV_NET_Model_state_dict'])
        self.actor.load_state_dict(checkpoint['ACTOR_Model_state_dict'])
        self.critic.load_state_dict(checkpoint['CRITIC_Model_state_dict'])
        self.hyper_parameter.ConfigImport()

    def ProbForward(self, state, action):
        state_feature = self.env_net(state)
        Mu, Sigma = self.actor(state_feature)
        distribution = torch.distributions.Normal(Mu, Sigma)
        logProb = distribution.log_prob(action)
        return logProb

    def ActionForward(self, state):
        state_feature = self.env_net(state)
        Mu, Sigma = self.actor(state_feature)
        distribution = torch.distributions.Normal(Mu, Sigma)
        action = distribution.sample().clamp(-2, 2)
        return action
    
    def ValueForward(self, state):
        state_feature = self.env_net(state)
        value = self.critic(state_feature)
        return value
    
    def TrainNet(self):
        for _ in range(self.hyper_parameter.update_num):
            select_data = self.data_store.LoadData(self.hyper_parameter.data_select)
            state = [tup for data in select_data for tup in data[0]]
            action = torch.tensor(numpy.array([data[1] for data in select_data]), dtype=torch.float32).to(self.device)
            reward = torch.tensor(numpy.array([data[2] for data in select_data]), dtype=torch.float32).to(self.device)
            next_state = [tup for data in select_data for tup in data[3]]
            action_prob = torch.tensor(numpy.array([data[4] for data in select_data]), dtype=torch.float32).to(self.device)

            reward = (reward - reward.mean()) / (reward.std() + 1e-5)
            with torch.no_grad():
                target_value = reward + 0.99 * self.ValueForward(next_state)
            advantage = (target_value - self.ValueForward(state)).detach()

            new_action_prob = self.ProbForward(state, action)
            ratio = torch.exp(new_action_prob - action_prob)
            L1 = ratio * advantage
            L2 = torch.clamp(ratio, 1 - self.hyper_parameter.clamp_value, 1 + self.hyper_parameter.clamp_value) * advantage
            actor_loss = -torch.min(L1, L2).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.sigma_netWork.parameters(), self.hyper_parameter.clip_value)
            torch.nn.utils.clip_grad_norm_(self.actor.mu_netWork.parameters(), self.hyper_parameter.clip_value)
            self.actor_optimizer.step()
            self.actor_loss.append(actor_loss.item())

            value_loss = self.lossFunction(self.ValueForward(state), target_value)
            self.critic_optimizer.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.hyper_parameter.clip_value)
            self.critic_optimizer.step()
            self.critic_loss.append(value_loss.item())

    def PlayGame(self, environment, epoch=0, role=0):
        environment.ResetEnviroment()
        state = environment.StateGet()
        over = False
        while not over:
            action = self.ActionForward(state)
            next_state, reward, truncated, over = environment.Step(action.item())
            self.data_store.SaveData((state, action.item(), reward, next_state, over))
            state = next_state
            self.reward.append(reward)
        
        sum_reward = sum(self.reward)
        if role == 1:   # Central类
            print(f'Epoch: {epoch}, ActorLoss: {sum(self.actor_loss)}, CriticLoss: {sum(self.critic_loss)}, Reward: {sum_reward}')
        self.Initialize()
        return sum_reward

# Worker类
class Worker:
    def __init__(self, trian_net:TrainNet) -> None:
        self.trian_net = trian_net
    
    def ResetNetState(self):
        self.trian_net.data_store.ClearData()

    def GenerateData(self, enviroment):
        self.trian_net.PlayGame(enviroment)
    
    def CommitData(self) -> list:
        return self.trian_net.data_store.data_store

    def UpdateNetPatameter(self, new_train_net:TrainNet):
        self.trian_net.actor.load_state_dict(new_train_net.actor.state_dict())
        self.trian_net.critic.load_state_dict(new_train_net.critic.state_dict())
        self.trian_net.env_net.load_state_dict(new_train_net.env_net.state_dict())

# Central类
class Central:
    def __init__(self, device) -> None:
        self.main_net = TrainNet(device)

    def GetData(self, data_list:list):
        for item in data_list:
            self.main_net.data_store.SaveData(item)
        
        while self.main_net.data_store.Length() > self.main_net.hyper_parameter.data_max:
            self.main_net.data_store.PopData()