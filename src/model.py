import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from timm.layers.adaptive_avgmax_pool import SelectAdaptivePool2d


#######################
#### EfficientNet
#######################
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
        self.aux_loss_features = args.aux_loss_features
        self.aux_loss_feature_outnum = args.aux_loss_feature_outnum
        
        self.training = training
        self.encoder = timm.create_model(args.model_name, pretrained=self.training,
                                          drop_path_rate=args.drop_path_rate)
        self.classifier_in_features = self.encoder.classifier.in_features
        self.features = nn.Sequential(*list(self.encoder.children())[:-2])
        self.GeM = GeM(p=args.gem_p)
        self.dropout_main = nn.ModuleList([
			nn.Dropout(args.dropout) for _ in range(5)
		]) #droupout augmentation
        self.linear_main = nn.Linear(self.classifier_in_features * 2, args.num_classes)

        if self.aux_loss_features is not None:
            self.aux_dropout = nn.ModuleList([nn.ModuleList([nn.Dropout(args.dropout) for _ in range(5)]) for _ in self.aux_loss_features])
            self.aux_linear = nn.ModuleList([nn.Linear(self.encoder.num_features, outnum) for outnum in self.aux_loss_feature_outnum])

        if args.use_metadata_num:
            self.block_1 = nn.Sequential(
                nn.Linear(args.use_metadata_num, self.classifier_in_features * 4),
                nn.BatchNorm1d(self.classifier_in_features * 4),
                nn.SiLU(),
                nn.Dropout(args.dropout),
            )
            self.block_2 = nn.Sequential(
                nn.Linear(self.classifier_in_features * 4, self.classifier_in_features),
                nn.BatchNorm1d(self.classifier_in_features),
                nn.SiLU(),
            )

    def forward(self, images, metadata):
        out = self.features(images)
        meta_out = self.block_1(metadata)
        meta_out = self.block_2(meta_out)
        out = self.GeM(out).flatten(1)
        out = torch.cat([out, meta_out], dim=1)
        if self.training:
            main_out = 0
            for i in range(len(self.dropout_main)):
                main_out += self.linear_main(self.dropout_main[i](out))
            main_out = main_out / len(self.dropout_main)

            aux_outs = []
            if self.aux_loss_features is not None:
                for aux_dropout, aux_linear in zip(self.aux_dropout, self.aux_linear):
                    out_aux = 0
                    for i in range(len(aux_dropout)):
                        out_aux += aux_linear(aux_dropout[i](out))
                    out_aux = out_aux / len(aux_dropout)
                    aux_outs.append(out_aux)
                return main_out, aux_outs
        else:
            main_out = self.linear_main(out)

        return main_out
    


#######################
#### Eva02
#######################
class CustomModelEva(nn.Module):
    def __init__(self, 
                 args,
                 training: bool = True, 
                 ):
        super(CustomModelEva, self).__init__()
        self.aux_loss_features = args.aux_loss_features
        self.aux_loss_feature_outnum = args.aux_loss_feature_outnum
        
        self.training = training
        self.encoder = timm.create_model(args.model_name, pretrained=self.training,  # Always use pretrained weights
                                          drop_path_rate=args.drop_path_rate)
        self.classifier_in_features = self.encoder.head.in_features
        self.encoder.head = nn.Identity()
        self.dropout_main = nn.ModuleList([
                                nn.Dropout(args.dropout) for _ in range(5)
                              ])  # Dropout augmentation
        self.linear_main = nn.Linear(self.classifier_in_features, args.num_classes)

        if self.aux_loss_features is not None:
            self.aux_dropout = nn.ModuleList([nn.ModuleList([nn.Dropout(args.dropout) for _ in range(5)]) for _ in self.aux_loss_features])
            self.aux_linear = nn.ModuleList([nn.Linear(self.encoder.num_features, outnum) for outnum in self.aux_loss_feature_outnum])

    def forward(self, images):
        out = self.encoder(images)
        if self.training:
            main_out = 0
            for i in range(len(self.dropout_main)):
                main_out += self.linear_main(self.dropout_main[i](out))
            main_out = main_out / len(self.dropout_main)

            aux_outs = []
            if self.aux_loss_features is not None:
                for aux_dropout, aux_linear in zip(self.aux_dropout, self.aux_linear):
                    out_aux = 0
                    for i in range(len(aux_dropout)):
                        out_aux += aux_linear(aux_dropout[i](out))
                    out_aux = out_aux / len(aux_dropout)
                    aux_outs.append(out_aux)
                return main_out, aux_outs
        else:
            main_out = self.linear_main(out)

        return main_out



#######################
#### ConvNext
#######################
class LayerNorm2d(nn.Module):
    def __init__(self, num_features):
        super(LayerNorm2d, self).__init__()
        self.layer_norm = nn.LayerNorm([num_features, 1, 1])

    def forward(self, x):
        return self.layer_norm(x)

class CustomConvNextModel(nn.Module):
    def __init__(self, args, training: bool = True):
        super(CustomConvNextModel, self).__init__()
        self.aux_loss_ratio = args.aux_loss_ratio
        self.training = training
        self.encoder = timm.create_model(args.model_name, pretrained=training,
                                         drop_path_rate=args.drop_path_rate)
        self.features = nn.Sequential(*list(self.encoder.children())[:-2])
        self.GAP = nn.AdaptiveAvgPool2d(1)
        self.layer_norm = LayerNorm2d(self.encoder.num_features)
        self.flatten = nn.Flatten()
        self.dropout_main = nn.ModuleList([nn.Dropout(args.dropout) for _ in range(5)])  # Dropout augmentation
        self.linear_main = nn.Linear(self.encoder.num_features, args.num_classes)

        if self.aux_loss_ratio is not None:
            self.dropout_aux = nn.ModuleList([nn.Dropout(args.dropout) for _ in range(5)])  # Dropout augmentation
            self.linear_aux = nn.Linear(self.encoder.num_features, 4)

    def forward(self, images):
        out = self.features(images)
        out = self.GAP(out)
        out = self.flatten(out)

        if self.training:
            main_out = 0
            for i in range(len(self.dropout_main)):
                main_out += self.linear_main(self.dropout_main[i](out))
            main_out = main_out / len(self.dropout_main)

            if self.aux_loss_ratio is not None:
                out_aux = 0
                for i in range(len(self.dropout_aux)):
                    out_aux += self.linear_aux(self.dropout_aux[i](out))
                out_aux = out_aux / len(self.dropout_aux)
                return main_out, out_aux     
        else:
            main_out = self.linear_main(out)
        
        return main_out
    


#######################
#### Swin
#######################
class CustomSwinModel(nn.Module):
    def __init__(self, args, training: bool = True):
        super(CustomSwinModel, self).__init__()
        self.aux_loss_features = args.aux_loss_features
        self.aux_loss_feature_outnum = args.aux_loss_feature_outnum

        self.training = training
        self.encoder = timm.create_model(args.model_name, pretrained=training,
                                         drop_path_rate=args.drop_path_rate)
        self.features = nn.Sequential(*list(self.encoder.children())[:-1])
        self.GAP = SelectAdaptivePool2d(pool_type='avg', input_fmt='NHWC', flatten=True)
        self.dropout_main = nn.ModuleList([nn.Dropout(args.dropout) for _ in range(5)])  # Dropout augmentation
        self.linear_main = nn.Linear(self.encoder.num_features, args.num_classes)

        if self.aux_loss_features is not None:
            self.aux_dropout = nn.ModuleList([nn.ModuleList([nn.Dropout(args.dropout) for _ in range(5)]) for _ in self.aux_loss_features])
            self.aux_linear = nn.ModuleList([nn.Linear(self.encoder.num_features, outnum) for outnum in self.aux_loss_feature_outnum])

    def forward(self, images):
        out = self.features(images)
        out = self.GAP(out)

        if self.training:
            main_out = 0
            for i in range(len(self.dropout_main)):
                main_out += self.linear_main(self.dropout_main[i](out))
            main_out = main_out / len(self.dropout_main)

            aux_outs = []
            if self.aux_loss_features is not None:
                for aux_dropout, aux_linear in zip(self.aux_dropout, self.aux_linear):
                    out_aux = 0
                    for i in range(len(aux_dropout)):
                        out_aux += aux_linear(aux_dropout[i](out))
                    out_aux = out_aux / len(aux_dropout)
                    aux_outs.append(out_aux)
                return main_out, aux_outs
        else:
            main_out = self.linear_main(out)

        return main_out
    


#######################
#### ResNet
####################### 
class CustomModelResNet(nn.Module):
    def __init__(self, 
                 args,
                 training: bool = True, 
                 ):
        super(CustomModelResNet, self).__init__()
        self.aux_loss_features = args.aux_loss_features
        self.aux_loss_feature_outnum = args.aux_loss_feature_outnum
        
        self.training = training
        self.encoder = timm.create_model(args.model_name, pretrained=self.training,
                                          drop_path_rate=args.drop_path_rate)
        self.features = nn.Sequential(*list(self.encoder.children())[:-2])

        self.classifier_in_features = self.encoder.fc.in_features
        self.GeM = GeM(p=args.gem_p)
        self.dropout_main = nn.ModuleList([
          nn.Dropout(args.dropout) for _ in range(5)
        ]) #droupout augmentation
        self.linear_main = nn.Linear(self.classifier_in_features, args.num_classes)

        if self.aux_loss_features is not None:
            self.aux_dropout = nn.ModuleList([nn.ModuleList([nn.Dropout(args.dropout) for _ in range(5)]) for _ in self.aux_loss_features])
            self.aux_linear = nn.ModuleList([nn.Linear(self.encoder.num_features, outnum) for outnum in self.aux_loss_feature_outnum])

    def forward(self, images):
        out = self.features(images)
        out = self.GeM(out).flatten(1)
        if self.training:
            main_out = 0
            for i in range(len(self.dropout_main)):
                main_out += self.linear_main(self.dropout_main[i](out))
            main_out = main_out / len(self.dropout_main)

            aux_outs = []
            if self.aux_loss_features is not None:
                for aux_dropout, aux_linear in zip(self.aux_dropout, self.aux_linear):
                    out_aux = 0
                    for i in range(len(aux_dropout)):
                        out_aux += aux_linear(aux_dropout[i](out))
                    out_aux = out_aux / len(aux_dropout)
                    aux_outs.append(out_aux)
                return main_out, aux_outs
        else:
            main_out = self.linear_main(out)

        return main_out