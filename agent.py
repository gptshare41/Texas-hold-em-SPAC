import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self, input_dim, hidden_dim=256):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return torch.sigmoid(self.fc3(x)) * 100  # X > 0 보장, 범위 조정

class PokerActor:
    def __init__(self, config):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actors = []
        input_dims = [
            3,  # nn1: [자신의 카드1, 자신의 카드2, 전체 팟(T)]
            7,  # nn2: [자신의 카드1, 자신의 카드2, 커뮤니티 카드1-3, 상대의 첫 번째 배팅 금액, 전체 팟(T)]
            9,  # nn3: [자신의 카드1, 자신의 카드2, 커뮤니티 카드1-4, 상대의 첫 번째 배팅 금액, 상대의 두 번째 배팅 금액, 전체 팟(T)]
            11, # nn4: [자신의 카드1, 자신의 카드2, 커뮤니티 카드1-5, 상대의 첫 번째 배팅 금액, 상대의 두 번째 배팅 금액, 상대의 세 번째 배팅 금액, 전체 팟(T)]
            4,  # nn5: [자신의 카드1, 자신의 카드2, 상대의 첫 번째 배팅 금액, 전체 팟(T)]
            8,  # nn6: [자신의 카드1, 자신의 카드2, 커뮤니티 카드1-3, 상대의 첫 번째 배팅 금액, 상대의 두 번째 배팅 금액, 전체 팟(T)]
            10, # nn7: [자신의 카드1, 자신의 카드2, 커뮤니티 카드1-4, 상대의 첫 번째 배팅 금액, 상대의 두 번째 배팅 금액, 상대의 세 번째 배팅 금액, 전체 팟(T)]
            12  # nn8: [자신의 카드1, 자신의 카드2, 커뮤니티 카드1-5, 상대의 첫 번째 배팅 금액, 상대의 두 번째 배팅 금액, 상대의 세 번째 배팅 금액, 상대의 네 번째 배팅 금액, 전체 팟(T)]
        ]
        for dim in input_dims:
            actor = Actor(dim, config['hidden_dim']).to(self.device)
            self.actors.append(actor)

    def get_action(self, state, nn_index):
        state_tensor = torch.tensor(state, dtype=torch.float).to(self.device)
        with torch.no_grad():
            x = self.actors[nn_index](state_tensor)
        return x.item()
