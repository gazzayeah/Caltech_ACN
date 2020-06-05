import torch.nn as nn
import torch.nn.functional as F


class LearningAgent(nn.Module):
    # regression model
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(LearningAgent, self).__init__()

        # architecture
        self.linear1 = nn.Linear(state_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)

        return x
