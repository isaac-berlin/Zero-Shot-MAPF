import torch
import torch.nn as nn


class ActorHybrid(nn.Module):
    def __init__(self, obs_spec, n_actions, hidden=128):
        super().__init__()
        h, w, channels = obs_spec["window"]
        vec_dim = obs_spec["vector"][0]

        self.cnn = nn.Sequential(
            nn.Conv2d(channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
        )

        with torch.no_grad():
            x = torch.zeros(1, channels, h, w)
            conv_out = self.cnn(x).view(1, -1).shape[1]

        self.cnn_fc = nn.Sequential(
            nn.LayerNorm(conv_out),
            nn.Linear(conv_out, hidden),
            nn.Tanh(),
        )

        self.knn_fc = nn.Sequential(
            nn.LayerNorm(vec_dim),
            nn.Linear(vec_dim, hidden),
            nn.Tanh(),
        )

        self.fusion_fc = nn.Sequential(
            nn.Linear(2 * hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, obs):
        vec_obs = obs["vector"]
        win_obs = obs["window"]

        win_x = win_obs.permute(0, 3, 1, 2)
        win_feat = self.cnn(win_x)
        win_feat = win_feat.reshape(win_feat.size(0), -1)
        win_out = self.cnn_fc(win_feat)

        knn_out = self.knn_fc(vec_obs)

        fusion_input = torch.cat([win_out, knn_out], dim=-1)
        return self.fusion_fc(fusion_input)


class ActorMLP(nn.Module):
    """Legacy actor kept only for compatibility with older checkpoints."""

    def __init__(self, obs_dim, n_actions, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(obs_dim),
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, obs):
        return self.net(obs)


class ActorCNN(nn.Module):
    """Used for obs_mode = window (CNN)."""

    def __init__(self, obs_shape, n_actions, hidden=128):
        super().__init__()
        h, w, channels = obs_shape

        self.cnn = nn.Sequential(
            nn.Conv2d(channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
        )

        with torch.no_grad():
            x = torch.zeros(1, channels, h, w)
            conv_out = self.cnn(x).view(1, -1).shape[1]

        self.fc = nn.Sequential(
            nn.LayerNorm(conv_out),
            nn.Linear(conv_out, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, obs):
        obs = obs.permute(0, 3, 1, 2)
        x = self.cnn(obs)
        x = x.reshape(x.size(0), -1)
        return self.fc(x)


class CentralCritic(nn.Module):
    """Central critic shared across all agents."""

    def __init__(self, state_shape, hidden=64):
        super().__init__()
        if len(state_shape) != 3:
            raise ValueError("CentralCritic expects state_shape=(C, H, W).")

        channels, _height, _width = state_shape

        self.cnn = nn.Sequential(
            nn.Conv2d(channels, 8, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )

        self.net = nn.Sequential(
            nn.LayerNorm(16),
            nn.Linear(16, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, state):
        if state.dim() == 3:
            state = state.unsqueeze(0)

        x = self.cnn(state)
        x = x.reshape(x.size(0), -1)
        return self.net(x).squeeze(-1)
