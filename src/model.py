import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1)*p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)
        
    def gem(self, x, p=3, eps=1e-6):
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)
        
    def __repr__(self):
        return self.__class__.__name__ + \
                '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + \
                ', ' + 'eps=' + str(self.eps) + ')'

    
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
        self.GeM = GeM()
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

    def forward(self, images):
        out = self.features(images)
        out = self.GeM(out)
        main_out = self.decoder(out.view(out.size(0), -1))
        
        if self.aux_loss_ratio is not None:
            out_aux = self.decoder_aux(out.view(out.size(0), -1))
            return main_out, out_aux
        else:
            return main_out