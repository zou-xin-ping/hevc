import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
def compute_cosine_sum(X, Y):
  X = X.reshape(X.shape[0], -1)
  Y = Y.reshape(Y.shape[0], -1)
  loss = F.cosine_similarity(X, Y, dim=-1)
  loss = (-1)*loss
  return torch.mean(loss)

class VGG16FeatureExtractor(nn.Module):
    def __init__(self):
        super(VGG16FeatureExtractor, self).__init__()
        self.vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features[:17] # 新版本的写法

    def forward(self, x):
        return self.vgg16(x)

vgg16_feature_extractor = VGG16FeatureExtractor()

import torch
import torchvision.transforms as transforms
from PIL import Image

# Load image
img = Image.open(r"H:\image_reloc\sppnet-pytorch-master\c992737_0.png")
img2 =Image.open(r'H:\image_reloc\sppnet-pytorch-master\c992737_0_0dot9.png')
# Preprocess image
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
img_tensor = transform(img).unsqueeze(0)
print(img_tensor.shape)
img_tensor2 = transform(img).unsqueeze(0)
print(img_tensor2.shape)
# Extract features
#features = vgg16_feature_extractor(img_tensor)
features_p = vgg16_feature_extractor(img_tensor)
features_d = vgg16_feature_extractor(img_tensor2)
print(features_p.shape)
print(features_d.shape)
cosine = compute_cosine_sum(features_p,features_d)
print(cosine)