"""
CNN-ViT Hybrid model architecture (ResNet50 + DeiT-Small fusion).

This must match the architecture used in training exactly, since it is used
to load the saved checkpoint's state_dict. If you change the training
architecture, update this file too.
"""
import torch
import torch.nn as nn
import timm


class CNNViTHybrid(nn.Module):
    def __init__(self, num_classes, cnn_model="resnet50", vit_model="deit_small_patch16_224"):
        super(CNNViTHybrid, self).__init__()

        self.cnn = timm.create_model(cnn_model, pretrained=False)
        if hasattr(self.cnn, "fc"):
            in_features_cnn = self.cnn.fc.in_features
            self.cnn.fc = nn.Identity()
        elif hasattr(self.cnn, "classifier"):
            in_features_cnn = self.cnn.classifier.in_features
            self.cnn.classifier = nn.Identity()
        else:
            in_features_cnn = 2048

        self.vit = timm.create_model(vit_model, pretrained=False)
        if hasattr(self.vit, "head"):
            in_features_vit = self.vit.head.in_features
            self.vit.head = nn.Identity()
        elif hasattr(self.vit, "fc"):
            in_features_vit = self.vit.fc.in_features
            self.vit.fc = nn.Identity()
        else:
            in_features_vit = 384

        fusion_dim = 512
        self.fusion = nn.Sequential(
            nn.Linear(in_features_cnn + in_features_vit, fusion_dim),
            nn.BatchNorm1d(fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

        self.num_classes = num_classes

    def forward(self, x):
        cnn_features = self.cnn(x)
        vit_features = self.vit(x)
        combined = torch.cat([cnn_features, vit_features], dim=1)
        fused = self.fusion(combined)
        output = self.classifier(fused)
        return output


def load_model(checkpoint_path, device="cpu"):
    """
    Loads a trained CNNViTHybrid model from a checkpoint saved by the
    training script. Returns (model, classes, config) where classes is the
    list of class names in the order the model was trained on, and config
    is the dict of training-time settings (image_size, cnn_model, etc).

    pretrained=False above is intentional: at inference time we are loading
    fine-tuned weights, not ImageNet weights, and downloading ImageNet
    weights here would be wasted bandwidth/time and could fail with no
    internet access at deployment time.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    classes = checkpoint["classes"]
    config = checkpoint.get("config", {})
    num_classes = config.get("num_classes", len(classes))
    cnn_model = config.get("cnn_model", "resnet50")
    vit_model = config.get("vit_model", "deit_small_patch16_224")

    model = CNNViTHybrid(num_classes=num_classes, cnn_model=cnn_model, vit_model=vit_model)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, classes, config
