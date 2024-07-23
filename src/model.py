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
                 args,
                 training: bool = True, 
                 ):
        super(CustomModel, self).__init__()
        self.aux_loss_ratio = args.aux_loss_ratio
        self.training = training
        self.encoder = timm.create_model(args.model_name, pretrained=self.training,
                                          drop_path_rate=args.drop_path_rate)
        self.features = nn.Sequential(*list(self.encoder.children())[:-2])
        self.GeM = GeM(p=args.gem_p)
        self.flatten = nn.Flatten()
        self.dropout_main = nn.ModuleList([
			nn.Dropout(args.dropout_rate) for _ in range(5)
		]) #droupout augmentation
        self.linear_main = nn.Linear(self.encoder.num_features, args.num_classes)

        if args.aux_loss_ratio is not None:
            self.decoder_aux = nn.Flatten()
            self.dropout_aux = nn.ModuleList([
			nn.Dropout(args.dropout_rate) for _ in range(5)
		]) #droupout augmentation
            self.linear_aux = nn.Linear(self.encoder.num_features, 4)

    def forward(self, images):
        out = self.features(images)
        out = self.flatten(out)
        out = self.GeM(out)
        if self.training:
            main_out=0
            for i in range(len(self.dropout_main)):
                main_out += self.target(self.dropout_main[i](out))
            main_out = main_out/len(self.dropout_main)
            if self.aux_loss_ratio is not None:
                out_aux=0
                for i in range(len(self.dropout_aux)):
                    out_aux += self.target(self.dropout_aux[i](out))
                out_aux = out_aux/len(self.dropout_aux)
                return main_out, out_aux
        else:
            main_out = self.linear_main(out)
        
        return main_out