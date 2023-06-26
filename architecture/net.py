import torchvision, torch
import torch.nn as nn

class ClassifierNet1(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_features: int,
        dropout:float=0.2, 
        network:str='resnet18',
        pretrained_weights: str="DEFAULT",
        track_grads: bool=True):
      
        super(ClassifierNet1, self).__init__()
        
        self.in_channels = in_channels
        self.out_features = out_features
        self.dropout = dropout
        self.network = network
        self.pretrained_weights = pretrained_weights
        self.track_grads = track_grads
        
        if self.in_channels != 3:
            self.rgb_channel_projector = nn.Conv2d(
                self.in_channels, 3, kernel_size=(3, 3), stride=1, padding=1)
            
        self.model = getattr(torchvision.models, self.network)(weights=self.pretrained_weights)
        
        named_modules = list(self.model.named_modules())
        self.dropout_layer = nn.Dropout(self.dropout)
        self.fc = nn.Sequential(
            nn.Linear(named_modules[-1][1].in_features, self.out_features),
            nn.Softmax(dim=-1)
        )
        setattr(self.model, named_modules[-1][0].split(".")[0], nn.Identity())
        
        for params in self.model.parameters():
            params.requires_grad_(self.track_grads)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            x = x.unsqueeze(dim=1)
        if self.in_channels != 3:
            x = self.rgb_channel_projector(x)
        output = self.model(x)
        try:
            output = self.dropout_layer(output).squeeze(2).squeeze(2)
        except IndexError:
            output = self.dropout_layer(output)
        output = self.fc(output)
        
        return output


class ClassificationNet2(nn.Module):
    def __init__(self, in_channels: int, input_dim: int, num_classes: int, dropout: float=0.2):
        super(ClassificationNet2, self).__init__()

        self.in_channels = in_channels
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.dropout = dropout

        self.layer1 = nn.Sequential(
            nn.Conv1d(self.in_channels, 128, kernel_size=2, stride=2),
            nn.BatchNorm1d(128),
            nn.Dropout(self.dropout),
        )

        self.layer2 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(self.dropout),
        )

        self.layer3 = nn.Sequential(
            nn.Conv1d(256, 512, kernel_size=3, stride=2),
            nn.ReLU(),
        )

        self.layer4 = nn.Sequential(
            nn.Linear(self._get_fc_size(), 128),
            nn.Dropout(self.dropout),
            nn.Linear(128, self.num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4 and x.shape[1] == 1:
            x = x.squeeze()
        output = self.layer1(x)
        output = self.layer2(output)
        output = self.layer3(output)
        output = output.reshape(output.shape[0], -1)
        output = self.layer4(output)
        return output


    def _get_fc_size(self) -> int: 
        x = torch.randn(1, self.in_channels, self.input_dim)
        output = self.layer3(self.layer2(self.layer1(x)))
        _, C, T = output.shape
        return C * T
