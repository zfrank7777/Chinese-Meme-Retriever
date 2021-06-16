from torchvision import models, transforms
from PIL import Image, ImageFont, ImageDraw
import random
import cv2
import torch
import torch.nn as nn
import numpy as np
import re
from numpy import linalg as LA
from tqdm import tqdm
import os
import pickle 


def read_cnn_featuers():
    with open('meme/cnn_features.pkl', 'rb') as f:
        features = pickle.load(f)
    return features

class Conv4(nn.Module):
    def __init__(self, model):
        super(Conv4, self).__init__()
        self.features = nn.Sequential(
            # stop at conv4
            *list(model.features.children())[:-3]
        )
    def forward(self, x):
        x = self.features(x)
        return x


class CNN:
    def __init__(self):
        model = models.alexnet(pretrained=True)
        # self.model = Conv4(model)
        self.model = model
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    def extract(self, img):
        img = self.transform(img).unsqueeze(0)
        with torch.no_grad():
            feature = self.model.forward(img).reshape(-1).squeeze()
        return feature.cpu().detach().numpy()


if __name__ == "__main__":
    cnn = CNN()
    file_folder = 'meme'
    features = {}
    for dirPath, dirNames, fileNames in os.walk(file_folder):
        for fn in tqdm(fileNames, total=len(fileNames)):
            if '.jpg' not in fn:
                continue
            if '0813' in fn:
                continue
            img_path = dirPath + '/' + fn
        
            img = Image.open(img_path).convert("RGB")

            feature = cnn.extract(img)
            features[img_path] = feature

    with open(os.path.join('meme', 'cnn_features.pkl'), 'wb') as f:
        pickle.dump(features, f)
