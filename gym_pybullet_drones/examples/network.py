import torch
import torch.nn as nn
import torch.nn.functional as F
import param

class Actor(nn.Module):
    def __init__(self, input_size, output_size):
        super(Actor, self).__init__()
        self.linear1 = nn.Linear(input_size, 64)
        self.linear2 = nn.Linear(64, 128)
        self.linear3 = nn.Linear(128, output_size)
        
    def forward(self, s):
        x = F.relu(self.linear1(s))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)

        return x
    
    def save_network(self, filename):
        torch.save(self.state_dict(), filename)

    def load_network(self, filename):
        self.load_state_dict(torch.load(filename))


class Critic(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, 64)
        self.linear2 = nn.Linear(64, 128)
        self.linear3 = nn.Linear(128, output_size)

    def forward(self, s, a):
        x = torch.cat([s, a], -1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)

        return x

    def save_network(self, filename):
        torch.save(self.state_dict(), filename)

    def load_network(self, filename):
        self.load_state_dict(torch.load(filename))

class Actor_GNN(nn.Module):
    def __init__(self, input_size, embedding_size, output_size):
        super().__init__()
        self.self_encoder_1 = nn.Linear(input_size*2, 32)
        self.self_encoder_2 = nn.Linear(32, embedding_size)
        self.neighbor_encoder_1 = nn.Linear(input_size, 32)
        self.neighbor_encoder_2 = nn.Linear(32, embedding_size)
        

        self.linear1 = nn.Linear(embedding_size*2, 64)
        self.linear2 = nn.Linear(64, 16)
        self.linear3 = nn.Linear(16, output_size)

    def forward(self, s):
        # 输入应该是(N+1)*input_size的张量，N是邻居节点数量
        # 获得N+1个嵌入
        x1 = s[0:1, :]
        x2 = s[2:,:]
        x3 = s[1:2,:]
        x1 = torch.cat([x1, x3], -1)
        x1 = F.relu(self.self_encoder_1(x1))
        x1 = torch.mean(self.self_encoder_2(x1), dim=0)
        x2 = F.relu(self.neighbor_encoder_1(x2))
        x2 = torch.mean(self.neighbor_encoder_2(x2), dim=0)
        x = torch.cat([x1, x2], -1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        
        return x

    def save_network(self, filename):
        torch.save(self.state_dict(), filename)

    def load_network(self, filename):
        self.load_state_dict(torch.load(filename))
    
class Critic_GNN(nn.Module):
    def __init__(self, input_size, embedding_size, output_size):
        super().__init__()
        self.self_encoder_1 = nn.Linear(input_size*2, 32)
        self.self_encoder_2 = nn.Linear(32, embedding_size)
        self.neighbor_encoder_1 = nn.Linear(input_size, 32)
        self.neighbor_encoder_2 = nn.Linear(32, embedding_size)

        self.linear1 = nn.Linear(embedding_size*2 + param.action_dim, 64)
        self.linear2 = nn.Linear(64, 16)
        self.linear3 = nn.Linear(16, output_size)
        
    def forward(self, s, a):
        x1 = s[0:1, :]
        x2 = s[2:,:]
        x3 = s[1:2, :]
        x1 = torch.cat([x1,x3], -1)
        x1 = F.relu(self.self_encoder_1(x1))
        x1 = torch.mean(self.self_encoder_2(x1), dim=0)
        x2 = F.relu(self.neighbor_encoder_1(x2))
        x2 = torch.mean(self.neighbor_encoder_2(x2), dim=0)
        x = torch.cat([x1, x2, a], -1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        
        return x

    def save_network(self, filename):
        torch.save(self.state_dict(), filename)

    def load_network(self, filename):
        self.load_state_dict(torch.load(filename))