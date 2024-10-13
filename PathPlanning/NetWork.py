import json
import numpy
import torch
import random
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

# 超参数类
class HyperParameters:
    def __init__(self):
        self.ConfigImport()
        self.GetScanDegree()

    def GetScanDegree(self):
        config_path = 'Config/TrainConfig.json'
        try:
            with open(config_path, 'r') as file:
                config = json.load(file)["EnviromentConfig"]
                self.scan_degree = config["ScanDegree"]
        except:
            self.scan_degree = 7.2
        return

    def ConfigExport(self) -> None:
        config_path = 'Config/TrainConfig.json'
        try:
            with open(config_path, 'r') as file:
                config = json.load(file)
        except:
            config = {}
        
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
            "TrainEpoch": self.train_epoch,
            "ScanDegree": self.scan_degree
        }

        with open(config_path, 'w') as file:
            json.dump(config, file, indent=4)
        return

    def ConfigImport(self) -> None:
        config_path = 'Config/TrainConfig.json'
        try:
            with open(config_path, 'r') as file:
                config = json.load(file)
                hyperparameter_config = config.get("HyperParameters", {})
            
            self.actor_lr = hyperparameter_config.get("ActorLearningRate", 2e-4)
            self.critic_lr = hyperparameter_config.get("CriticLearningRate", 5e-4)
            self.gamma = hyperparameter_config.get("DiscountRate", 0.99)
            self.data_select = hyperparameter_config.get("DataSelect", 1000)
            self.update_num = hyperparameter_config.get("UpdateNumber", 1000)
            self.clamp_value = hyperparameter_config.get("ClampValue", 0.2)
            self.clip_value = hyperparameter_config.get("ClipValue", 0.2)
            self.data_max = hyperparameter_config.get("DataMax", 100000)
            self.worker_num = hyperparameter_config.get("WorkerNumber", 4)
            self.train_epoch = hyperparameter_config.get("TrainEpoch", 10000)
            self.scan_degree = hyperparameter_config.get("ScanDegree", 7.2)
        except:
            return
        return

# 状态特征类
class StateFeature(torch.nn.Module):
    def __init__(self, device:str, in_dim:int) -> None:
        '''
        :param device: "cuda", "cpu" \n
        '''
        super(StateFeature,self).__init__()
        self.img_netWork = torch.nn.Sequential(
            torch.nn.Linear(in_dim, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 1)
        ).to(device)
        self.device = device

    def forward(self, batch_states):
        state_maps = torch.tensor(numpy.array([state[0] for state in batch_states]), dtype=torch.float32).to(self.device)
        degree_feature = torch.tensor(numpy.array([state[1] for state in batch_states]), dtype=torch.float32).reshape(-1,1).to(self.device)
        env_states = self.img_netWork(state_maps).reshape(-1,1)

        env_features = torch.cat((env_states, degree_feature), dim=1)
        # print(env_states.shape, degree_feature.shape,env_features.shape)
        return env_features

# Actor类
class Actor(torch.nn.Module):
    def __init__(self, device:str) -> None:
        super(Actor,self).__init__()
        self.sigma_netWork = torch.nn.Sequential(
            torch.nn.Linear(2, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 1),
            torch.nn.Softplus()
        ).to(device)

        self.mu_netWork = torch.nn.Sequential(
            torch.nn.Linear(2, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 1),
            torch.nn.Tanh()
        ).to(device)

    def forward(self, env_feature):
        Mu = self.mu_netWork(env_feature) * 175
        Sigma = self.sigma_netWork(env_feature)
        Sigma = torch.clamp(Sigma, 0.1, 100)
        return Mu, Sigma

# Critic类
class Critic(torch.nn.Module):
    def __init__(self, device:str) -> None:
        super(Critic,self).__init__()
        self.value_netWork = torch.nn.Sequential(
            torch.nn.Linear(2, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 1)
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
        self.env_net = StateFeature(device, int(360/self.hyper_parameter.scan_degree))
        self.actor = Actor(device)
        self.new_actor = Actor(device)
        self.critic = Critic(device)
        self.data_store = Store()

        self.lossFunction = torch.nn.MSELoss()
        self.actor_optimizer = torch.optim.Adam(list(self.actor.parameters()) + list(self.env_net.parameters()), lr=self.hyper_parameter.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.hyper_parameter.critic_lr)
        self.InitialParameter()
        self.Initialize()

    def Initialize(self):
        self.reward = []
        self.actor_loss = []
        self.critic_loss = []

    def InitialParameter(self):
        self.new_actor.load_state_dict(self.actor.state_dict())

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
        logProb = distribution.log_prob(action).reshape(-1,1)
        return logProb
    
    def NewProbForward(self, state, action):
        state_feature = self.env_net(state)
        Mu, Sigma = self.new_actor(state_feature)
        distribution = torch.distributions.Normal(Mu, Sigma)
        logProb = distribution.log_prob(action).reshape(-1, 1)
        return logProb

    def ActionForward(self, state):
        state_feature = self.env_net(state)
        Mu, Sigma = self.actor(state_feature)
        distribution = torch.distributions.Normal(Mu, Sigma)
        action = distribution.sample().clamp(-175, 175) + random.randint(-5,5)
        return action
    
    def ValueForward(self, state):
        state_feature = self.env_net(state)
        value = self.critic(state_feature)
        return value
    
    def UpdateTargetNetWork(self, TargetNetWork, NetWork):
        TargetNetWork.load_state_dict(NetWork.state_dict())

    def TrainNet(self):
        select_data = self.data_store.LoadData(self.hyper_parameter.data_max)
        state = [tup for data in select_data for tup in data[0]]
        action = torch.tensor(numpy.array([data[1] for data in select_data]), dtype=torch.float32).reshape(-1,1).to(self.device)
        reward = torch.tensor(numpy.array([data[2] for data in select_data]), dtype=torch.float32).reshape(-1,1).to(self.device)
        next_state = [tup for data in select_data for tup in data[3]]
        over = torch.tensor(numpy.array([data[4] for data in select_data]), dtype=torch.float32).reshape(-1,1).to(self.device)
        action_prob = torch.tensor([data[5] for data in select_data], dtype=torch.float32).reshape(-1,1).to(self.device)

        reward = (reward - reward.mean()) / (reward.std() + 1e-5)
        with torch.no_grad():
            target_value = reward + self.hyper_parameter.gamma * self.ValueForward(next_state).reshape(-1,1) * (1 - over)
        advantage = (target_value - self.ValueForward(state).reshape(-1,1)).detach()

        for _ in range(self.hyper_parameter.update_num):
            for index in BatchSampler(SubsetRandomSampler(range(len(select_data))), int(self.hyper_parameter.data_select), False):
                state_list = [state[idx] for idx in index]

                new_action_prob = self.NewProbForward(state_list, action[index])
                ratio = torch.exp(new_action_prob - action_prob[index])
                # print(target_value.shape,advantage.shape,new_action_prob.shape,ratio.shape,action_prob.shape)
                L1 = ratio * advantage[index]
                L2 = torch.clamp(ratio, 1 - self.hyper_parameter.clamp_value, 1 + self.hyper_parameter.clamp_value) * advantage[index]
                actor_loss = -torch.min(L1, L2).mean()
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.sigma_netWork.parameters(), self.hyper_parameter.clip_value)
                torch.nn.utils.clip_grad_norm_(self.actor.mu_netWork.parameters(), self.hyper_parameter.clip_value)
                self.actor_optimizer.step()
                self.actor_loss.append(actor_loss.item())
            
                current_value = self.ValueForward(state_list).reshape(-1,1)
                value_loss = self.lossFunction(current_value, target_value[index])
                self.critic_optimizer.zero_grad()
                value_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.hyper_parameter.clip_value)
                self.critic_optimizer.step()
                self.critic_loss.append(value_loss.item())
            
                # for param in self.actor.parameters():
                #     if param.grad is not None:
                #         print(f'Gradient: {param.grad}')
                # for param in self.env_net.parameters():
                #     if param.grad is not None:
                #         print(f'Gradient: {param.grad}')
                # for param in self.critic.parameters():
                #     if param.grad is not None:
                #         print(f'Gradient: {param.grad}')
        self.UpdateTargetNetWork(self.actor, self.new_actor)
        self.data_store.ClearData()

    def PlayGame(self, environment, epoch=0):
        environment.ResetEnviroment(1,"GlobalPic_a")
        state = environment.StateGet()
        over = False
        while not over:
            action = self.ActionForward(state)
            action_prob = self.ProbForward(state, action)
            next_state, reward, truncated, over = environment.Step(action.item())
            over = over or truncated
            self.data_store.SaveData((state, action.item(), reward, next_state, over, action_prob))

            if self.data_store.Length() >= self.hyper_parameter.data_max:
                self.TrainNet()

            state = next_state
            self.reward.append(reward)
            sum_reward = sum(self.reward)

        if (epoch + 1) % 100 == 0:
            print(f'Epoch: {epoch}, ActorLoss: {sum(self.actor_loss)}, CriticLoss: {sum(self.critic_loss)}, Reward: {sum_reward}')
        self.Initialize()
        return sum_reward