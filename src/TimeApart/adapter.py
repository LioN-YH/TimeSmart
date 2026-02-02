import torch
import torch.nn as nn

class Adapter(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1):
        """
        Adapter module to project features from the backbone to the task-specific space.
        Structure: Linear -> GELU -> Dropout -> Linear
        """
        super(Adapter, self).__init__()
        self.down_project = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.up_project = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x: [B, input_dim]
        x = self.down_project(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.up_project(x)
        return x
