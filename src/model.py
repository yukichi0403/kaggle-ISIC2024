import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class CustomModel(nn.Module):
    def __init__(self, 
                 model_name, 
                 num_classes: int = 0, # write the number of classes
                 pretrained: bool = True, 
                 aux_loss_ratio: float = None, 
                 dropout_rate: float = 0):
        super(CustomModel, self).__init__()
        self.aux_loss_ratio = aux_loss_ratio
        self.encoder = timm.create_model(model_name, pretrained=pretrained,
                                          drop_path_rate=dropout_rate)
        self.features = nn.Sequential(*list(self.encoder.children())[:-2])
        self.GAP = nn.AdaptiveAvgPool2d(1)
        self.decoder = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.encoder.num_features, num_classes)
        )
        if aux_loss_ratio is not None:
            self.decoder_aux = nn.Sequential(
                nn.Flatten(),
                nn.Dropout(dropout_rate),
                nn.Linear(self.encoder.num_features, 4)
                )
        
    def expand_dims(self, images):
        # Expand dims to [B, H, W, 3]
        #images = images.unsqueeze(1).expand(-1, 3, -1, -1) #if you want to add a channel dimension
        return images

    def forward(self, images):
        #images = self.expand_dims(images) #if you want to add a channel dimension
        out = self.features(images)
        out = self.GAP(out)
        main_out = self.decoder(out.view(out.size(0), -1))
        
        if self.aux_loss_ratio is not None:
            out_aux = self.decoder_aux(out.view(out.size(0), -1))
            return main_out, out_aux
        else:
            return main_out