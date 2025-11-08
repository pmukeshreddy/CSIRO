class BiomassModel(nn.Module):
    def __init__(self,backbone_name="tf_efficientnetv2_m",num_metadata_features=9,num_targets=5,pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(backbone_name,pretrained=pretrained,num_classes=0,global_pool="avg")
        with torch.no_grad():
            dummy_img = torch.randn(1, 3, config.IMG_SIZE, config.IMG_SIZE)
            backbone_dim = self.backbone(dummy_img).shape[1]
        self.metadata_encoder = nn.Sequential(
            nn.Linear(num_metadata_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.fusion = CrossAttentionFusion(img_dim=backbone_dim,meta_dim=256,hidden_dim=512,num_heads=8)

        self.predictor = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_targets)
        )

    def forward(self,image,metadata):
        img_features = self.backbone(image)
        meta_features = self.metadata_encoder(metadata)

        fussed = self.fusion(img_features,meta_features)

        outputs = self.predictor(fussed)

        return outputs
