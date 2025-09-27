import torch
import torch.nn as nn
    

class CN_FC_1(nn.Module):
    def __init__(self):
        super(CN_FC_1, self).__init__()
        self.conv_surround = nn.Sequential(
            nn.Conv2d(7, 16, 4),
            nn.Conv2d(16, 32, 4),
            nn.Flatten()
        )
        self.conv_ego = nn.Sequential(
            nn.Conv2d(1, 16, 4),
            nn.Conv2d(16, 8, 4),
            nn.Flatten()
        )
        self.fc = nn.Sequential(
            nn.Linear(90, 50),
            nn.ReLU(),
            nn.Linear(50, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            
        )
        self.out = torch.nn.Linear(16, 1)
    
    def forward(self, ego_dop_input, sur_dop_input, ego_vector_input):
        CN_ego_output = self.conv_ego(ego_dop_input)
        CN_surround_output = self.conv_surround(sur_dop_input)
        
        CN_output = torch.cat([CN_ego_output, CN_surround_output, ego_vector_input], dim=1)
        output = self.fc(CN_output)
        score = self.out(output)
        score = torch.clamp(score, 0, 2)
        # score[score < 0.66] = 0
        # score[(score >= 0.66) & (score < 1.33)] = 1
        # score[score >= 1.33] = 2

        return score


class CN_FC(nn.Module):
    def __init__(self, flag_soft=True):
        super(CN_FC, self).__init__()
        self.conv_ego = nn.Sequential(
            nn.Conv2d(1, 16, 4),
            nn.Conv2d(16, 8, 4),
            nn.Flatten()
        )
        
        self.conv_surround = nn.Sequential(
            nn.Conv2d(7, 16, 4),
            nn.Conv2d(16, 32, 4),
            nn.Flatten()
        )

        self.fc = nn.Sequential(
            nn.Linear(90, 50),
            nn.ReLU(),
            nn.Linear(50, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
        )
        self.out = torch.nn.Linear(16, 3)
        self.flag_soft = flag_soft
    
    def forward(self, ego_dop_input, sur_dop_input, ego_vector_input, flag_before_soft=False):
        CN_ego_output = self.conv_ego(ego_dop_input)
        CN_surround_output = self.conv_surround(sur_dop_input)
        
        CN_output = torch.cat([CN_ego_output, CN_surround_output, ego_vector_input], dim=1)
        output = self.fc(CN_output)
        score = self.out(output)
        if self.flag_soft:
            score = torch.softmax(score, -1)
        return score

