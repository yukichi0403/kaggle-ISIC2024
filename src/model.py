import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from timm.layers.adaptive_avgmax_pool import SelectAdaptivePool2d



class AttentionFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim=dim, num_heads=8)
        
    def forward(self, image_features, metadata_features):
        # image_features: [batch_size, dim]
        # metadata_features: [batch_size, dim]
        
        # Reshape for attention: [seq_len, batch_size, dim]
        image_features = image_features.unsqueeze(0)
        metadata_features = metadata_features.unsqueeze(0)
        
        # Perform attention
        fused_features, _ = self.attention(image_features, metadata_features, metadata_features)
        
        # Reshape back: [batch_size, dim]
        return fused_features.squeeze(0)
    


class MultiheadAttentionMetadataEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, num_layers, dropout=0.1):
        super(MultiheadAttentionMetadataEncoder, self).__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.pos_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
            for _ in range(num_layers)
        ])
        self.norm_layers = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        self.final_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        # x shape: (batch_size, input_dim)
        x = self.input_proj(x)  # shape: (batch_size, hidden_dim)
        x = x.unsqueeze(1)  # Add sequence dimension: (batch_size, 1, hidden_dim)
        
        # 位置エンコーディングを各バッチに適用
        batch_size = x.size(0)
        pos = self.pos_encoder(torch.arange(1, device=x.device).float().unsqueeze(0).expand(batch_size, -1))
        pos = pos.unsqueeze(1)  # shape: (batch_size, 1, hidden_dim)
        x = x + pos
        
        for layer, norm in zip(self.layers, self.norm_layers):
            residual = x
            x = norm(x)
            x, _ = layer(x, x, x)
            x = self.dropout(x)
            x = residual + x
        
        x = x.squeeze(1)  # Remove sequence dimension: (batch_size, hidden_dim)
        x = self.output_proj(x)
        x = self.final_norm(x)
        return x



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
        self.use_metadata = args.use_metadata_num is not None and args.use_metadata_num > 0
        
        if self.use_metadata:
            self.linear_main = nn.Linear(self.classifier_in_features * 2, args.num_classes)
        else:
            self.linear_main = nn.Linear(self.classifier_in_features, args.num_classes)

        if self.aux_loss_features is not None:
            self.aux_dropout = nn.ModuleList([nn.ModuleList([nn.Dropout(args.dropout) for _ in range(5)]) for _ in self.aux_loss_features])
            self.aux_linear = nn.ModuleList([nn.Linear(self.encoder.num_features, outnum) for outnum in self.aux_loss_feature_outnum])

        if self.use_metadata:
            self.block_1 = nn.Sequential(
                nn.Linear(args.use_metadata_num, self.classifier_in_features * 4),
                nn.BatchNorm1d(self.classifier_in_features * 4),
                nn.SiLU(),
                nn.Dropout(args.dropout),
            )
            self.block_2 = nn.Sequential(
                nn.Linear(self.classifier_in_features * 4, self.classifier_in_features * 2),
                nn.BatchNorm1d(self.classifier_in_features * 2),
                nn.SiLU(),
                nn.Dropout(args.dropout),
            )
            self.block_3 = nn.Sequential(
                nn.Linear(self.classifier_in_features * 2, self.classifier_in_features),
                nn.BatchNorm1d(self.classifier_in_features),
                nn.SiLU(),
            )

    def forward(self, images, metadata=None):
        out = self.features(images)
        out = self.GeM(out).flatten(1)
        
        if self.use_metadata and metadata is not None:
            meta_out = self.block_1(metadata)
            meta_out = self.block_2(meta_out)
            meta_out = self.block_3(meta_out)
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

        self.use_metadata = args.use_metadata_num is not None and args.use_metadata_num > 0
        if self.use_metadata:
            self.linear_main = nn.Linear(self.encoder.num_features * 2, args.num_classes)
            self.block_1 = nn.Sequential(
                nn.Linear(args.use_metadata_num, self.encoder.num_features * 4),
                nn.BatchNorm1d(self.encoder.num_features * 4),
                nn.SiLU(),
                nn.Dropout(args.dropout),
            )
            self.block_2 = nn.Sequential(
                nn.Linear(self.encoder.num_features * 4, self.encoder.num_features * 2),
                nn.BatchNorm1d(self.encoder.num_features * 2),
                nn.SiLU(),
                nn.Dropout(args.dropout),
            )
            self.block_3 = nn.Sequential(
                nn.Linear(self.encoder.num_features * 2, self.encoder.num_features),
                nn.BatchNorm1d(self.encoder.num_features),
                nn.SiLU(),
            )
        else:
            self.linear_main = nn.Linear(self.encoder.num_features, args.num_classes)

    def forward(self, images, metadata=None):
        out = self.encoder(images)

        if self.use_metadata and metadata is not None:
            meta_out = self.block_1(metadata)
            meta_out = self.block_2(meta_out)
            meta_out = self.block_3(meta_out)
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
#### ConvNext
#######################
class LayerNorm2d(nn.Module):
    def __init__(self, num_features):
        super(LayerNorm2d, self).__init__()
        self.layer_norm = nn.LayerNorm([num_features, 1, 1])

    def forward(self, x):
        return self.layer_norm(x)

class CustomConvEdgeNextModel(nn.Module):
    def __init__(self, args, training: bool = True):
        super(CustomConvEdgeNextModel, self).__init__()
        self.aux_loss_ratio = args.aux_loss_ratio
        self.training = training
        self.encoder = timm.create_model(args.model_name, pretrained=training,
                                         drop_path_rate=args.drop_path_rate)
        self.features = nn.Sequential(*list(self.encoder.children())[:-2])
        self.GAP = nn.AdaptiveAvgPool2d(1)
        self.layer_norm = LayerNorm2d(self.encoder.num_features)
        self.flatten = nn.Flatten()
        self.dropout_main = nn.ModuleList([nn.Dropout(args.dropout) for _ in range(5)])  # Dropout augmentation

        if self.aux_loss_ratio is not None:
            self.dropout_aux = nn.ModuleList([nn.Dropout(args.dropout) for _ in range(5)])  # Dropout augmentation
            self.linear_aux = nn.Linear(self.encoder.num_features, 4)
        
        self.use_metadata = args.use_metadata_num is not None and args.use_metadata_num > 0
        if self.use_metadata:
            if args.metadata_head_type == "linear":
                self.metadata_encoder = nn.Sequential(
                    nn.Linear(args.use_metadata_num, args.metadata_dim),
                    nn.BatchNorm1d(args.metadata_dim),
                    nn.SiLU(),
                    nn.Dropout(0.3),
                    nn.Linear(args.metadata_dim, self.encoder.num_features),
                    nn.BatchNorm1d(self.encoder.num_features),
                    nn.SiLU(),
                )
            elif args.metadata_head_type == "attention":
                self.metadata_encoder = MultiheadAttentionMetadataEncoder(args.use_metadata_num, args.metadata_dim, 
                                                                          num_heads=8, num_layers=2, dropout=0.1)
            
            self.fusion = args.fusion_method  # New argument for fusion method
            if self.fusion == 'concat':
                self.linear_main = nn.Linear(self.encoder.num_features * 2, args.num_classes)
            elif self.fusion == 'gated':
                self.gate = nn.Sequential(
                    nn.Linear(self.encoder.num_features * 2, self.encoder.num_features),
                    nn.Sigmoid()
                )
                self.linear_main = nn.Linear(self.encoder.num_features, args.num_classes)
            elif self.fusion == 'attention':
                self.attention_fusion = AttentionFusion(dim=self.encoder.num_features)
                self.linear_main = nn.Linear(self.encoder.num_features, args.num_classes)
            else:
                self.linear_main = nn.Linear(self.encoder.num_features, args.num_classes)
        else:
            self.linear_main = nn.Linear(self.encoder.num_features, args.num_classes)

    def forward(self, images, metadata=None):
        image_features = self.features(images)
        image_features = self.GAP(image_features)
        image_features = self.layer_norm(image_features)
        image_features = self.flatten(image_features)

        if self.use_metadata and metadata is not None:
            metadata_features = self.metadata_encoder(metadata)
            
            if self.fusion == 'concat':
                fused_features = torch.cat([image_features, metadata_features], dim=1)
            elif self.fusion == 'gated':
                concat_features = torch.cat([image_features, metadata_features], dim=1)
                gate = self.gate(concat_features)
                fused_features = image_features * gate
            elif self.fusion == 'attention':
                fused_features = self.attention_fusion(image_features, metadata_features)
            else:
                fused_features = image_features
        else:
            fused_features = image_features
        if self.training:
            main_out = 0
            for i in range(len(self.dropout_main)):
                main_out += self.linear_main(self.dropout_main[i](fused_features))
            main_out = main_out / len(self.dropout_main)

            if self.aux_loss_ratio is not None:
                out_aux = 0
                for i in range(len(self.dropout_aux)):
                    out_aux += self.linear_aux(self.dropout_aux[i](fused_features))
                out_aux = out_aux / len(self.dropout_aux)
                return main_out, out_aux     
        else:
            main_out = self.linear_main(fused_features)
        
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
        self.dropout_main = nn.ModuleList([nn.Dropout(args.dropout) for _ in range(5)])
        
        self.use_metadata = args.use_metadata_num is not None and args.use_metadata_num > 0
        if self.use_metadata:
            if args.metadata_head_type == "linear":
                self.metadata_encoder = nn.Sequential(
                    nn.Linear(args.use_metadata_num, args.metadata_dim),
                    nn.BatchNorm1d(args.metadata_dim),
                    nn.SiLU(),
                    nn.Dropout(0.3),
                    nn.Linear(args.metadata_dim, self.encoder.num_features),
                    nn.BatchNorm1d(self.encoder.num_features),
                    nn.SiLU(),
                )
            elif args.metadata_head_type == "attention":
                self.metadata_encoder = MultiheadAttentionMetadataEncoder(args.use_metadata_num, args.metadata_dim, 
                                                                          num_heads=8, num_layers=2, dropout=0.1)
            
            self.fusion = args.fusion_method  # New argument for fusion method
            if self.fusion == 'concat':
                self.linear_main = nn.Linear(self.encoder.num_features * 2, args.num_classes)
            elif self.fusion == 'gated':
                self.gate = nn.Sequential(
                    nn.Linear(self.encoder.num_features * 2, self.encoder.num_features),
                    nn.Sigmoid()
                )
                self.linear_main = nn.Linear(self.encoder.num_features, args.num_classes)
            elif self.fusion == 'attention':
                self.attention_fusion = AttentionFusion(dim=self.encoder.num_features)
                self.linear_main = nn.Linear(self.encoder.num_features, args.num_classes)
            else:
                self.linear_main = nn.Linear(self.encoder.num_features, args.num_classes)
        else:
            self.linear_main = nn.Linear(self.encoder.num_features, args.num_classes)

        if self.aux_loss_features is not None:
            self.aux_dropout = nn.ModuleList([nn.ModuleList([nn.Dropout(args.dropout) for _ in range(5)]) for _ in self.aux_loss_features])
            self.aux_linear = nn.ModuleList([nn.Linear(self.encoder.num_features, outnum) for outnum in self.aux_loss_feature_outnum])

    def forward(self, images, metadata=None):
        image_features = self.features(images)
        image_features = self.GAP(image_features)

        if self.use_metadata and metadata is not None:
            metadata_features = self.metadata_encoder(metadata)
            
            if self.fusion == 'concat':
                fused_features = torch.cat([image_features, metadata_features], dim=1)
            elif self.fusion == 'gated':
                concat_features = torch.cat([image_features, metadata_features], dim=1)
                gate = self.gate(concat_features)
                fused_features = image_features * gate
            elif self.fusion == 'attention':
                fused_features = self.attention_fusion(image_features, metadata_features)
            else:
                fused_features = image_features
        else:
            fused_features = image_features

        if self.training:
            main_out = 0
            for i in range(len(self.dropout_main)):
                main_out += self.linear_main(self.dropout_main[i](fused_features))
            main_out = main_out / len(self.dropout_main)

            aux_outs = []
            if self.aux_loss_features is not None:
                for aux_dropout, aux_linear in zip(self.aux_dropout, self.aux_linear):
                    out_aux = 0
                    for i in range(len(aux_dropout)):
                        out_aux += aux_linear(aux_dropout[i](fused_features))
                    out_aux = out_aux / len(aux_dropout)
                    aux_outs.append(out_aux)
                return main_out, aux_outs
        else:
            main_out = self.linear_main(fused_features)

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



#######################
#### CoatNet
####################### 
class CustomCoatnetModel(nn.Module):
    def __init__(self, args, training: bool = True):
        super(CustomCoatnetModel, self).__init__()
        self.aux_loss_features = args.aux_loss_features
        self.aux_loss_feature_outnum = args.aux_loss_feature_outnum

        self.training = training
        self.encoder = timm.create_model(args.model_name, pretrained=training,
                                         drop_path_rate=args.drop_path_rate)
        self.features = nn.Sequential(*list(self.encoder.children())[:-1])  # Remove the final classification layer
        self.GeM = GeM(p=args.gem_p)
        self.dropout_main = nn.ModuleList([nn.Dropout(args.dropout) for _ in range(5)])  # Dropout augmentation
        
        self.use_metadata = args.use_metadata_num is not None and args.use_metadata_num > 0
        if self.use_metadata:
            self.linear_main = nn.Linear(768 * 2, args.num_classes)
            self.block_1 = nn.Sequential(
                nn.Linear(args.use_metadata_num, 768 * 4),
                nn.BatchNorm1d(768 * 4),
                nn.SiLU(),
                nn.Dropout(args.dropout),
            )
            self.block_2 = nn.Sequential(
                nn.Linear(768 * 4, 768 * 2),
                nn.BatchNorm1d(768 * 2),
                nn.SiLU(),
                nn.Dropout(args.dropout),
            )
            self.block_3 = nn.Sequential(
                nn.Linear(768 * 2, 768),
                nn.BatchNorm1d(768),
                nn.SiLU(),
            )
        else:
            self.linear_main = nn.Linear(768, args.num_classes)

        if self.aux_loss_features is not None:
            self.aux_dropout = nn.ModuleList([nn.ModuleList([nn.Dropout(args.dropout) for _ in range(5)]) for _ in self.aux_loss_features])
            self.aux_linear = nn.ModuleList([nn.Linear(768, outnum) for outnum in self.aux_loss_feature_outnum])

    def forward(self, images, metadata=None):
        out = self.features(images)
        out = self.GeM(out).flatten(1)

        if self.use_metadata and metadata is not None:
            meta_out = self.block_1(metadata)
            meta_out = self.block_2(meta_out)
            meta_out = self.block_3(meta_out)
            out = torch.cat([out, meta_out], dim=1)

        if self.training:
            main_out = sum(self.linear_main(dropout(out)) for dropout in self.dropout_main) / len(self.dropout_main)

            aux_outs = []
            if self.aux_loss_features is not None:
                for aux_dropout, aux_linear in zip(self.aux_dropout, self.aux_linear):
                    out_aux = sum(aux_linear(dropout(out)) for dropout in aux_dropout) / len(aux_dropout)
                    aux_outs.append(out_aux)
                return main_out, aux_outs
        else:
            main_out = self.linear_main(out)

        return main_out
