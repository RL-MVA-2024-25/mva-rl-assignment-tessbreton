from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
import torch
import torch.nn as nn

env = TimeLimit(env=HIVPatient(domain_randomization=True), max_episode_steps=200)
    
class ProjectAgent:
    def __init__(self, checkpoint_path='model.pt'):
        self.checkpoint_path = checkpoint_path

    def act(self, observation):
        device = "cuda" if next(self.model.parameters()).is_cuda else "cpu"
        with torch.no_grad():
            Q = self.model(torch.Tensor(observation).unsqueeze(0).to(device))
            return torch.argmax(Q).item()

    def load(self):
        device = torch.device('cpu')
        self.model = self.DQN(device)
        self.model.load_state_dict(torch.load(self.checkpoint_path, map_location=device))
        self.model.eval() 
    
    def DQN(self, device, hidden_dim=256):
        n_states = env.observation_space.shape[0]
        n_actions = env.action_space.n 
        network = torch.nn.Sequential(
                nn.Linear(n_states, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(), 
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(), 
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(), 
                nn.Linear(hidden_dim, hidden_dim), 
                nn.ReLU(),
                nn.Linear(hidden_dim, n_actions)
            ).to(device)
        return network
    
    def greedy_action(self, network, state):
        device = "cuda" if next(network.parameters()).is_cuda else "cpu"
        with torch.no_grad():
            Q = network(torch.Tensor(state).unsqueeze(0).to(device))
            return torch.argmax(Q).item()
        
    def save(self, path):
        pass