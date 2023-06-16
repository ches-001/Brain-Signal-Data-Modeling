import torchvision
import torch.nn as nn

class ClassifierNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_features: int,
        dropout:float=0.2, 
        network:str='resnet18',
        pretrained_weights: str="DEFAULT",
        track_grads: bool=True):
      
        super(ClassifierNet, self).__init__()
        
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
            #nn.GELU(),
            nn.Linear(named_modules[-1][1].in_features, self.out_features),
            nn.Softmax(dim=-1)
        )
        setattr(self.model, named_modules[-1][0].split(".")[0], nn.Identity())
        
        for params in self.model.parameters():
            params.requires_grad_(self.track_grads)
        
    def forward(self, x):
        if self.in_channels != 3:
            x = self.rgb_channel_projector(x)
        output = self.model(x)
        try:
            output = self.dropout_layer(output).squeeze(2).squeeze(2)
        except IndexError:
            output = self.dropout_layer(output)
        output = self.fc(output)
        
        return output
