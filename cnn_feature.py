from torchvision import models, transforms
from PIL import Image, ImageFont, ImageDraw
import random
import cv2
import torch
import torch.nn as nn
import numpy as np
import re
from numpy import linalg as LA


def pruning(raw_sentence):
    #sentence = re.sub('[A-Za-z0-9]+', '', raw_sentence)
    rule = re.compile(u"[^\u4e00-\u9fa5]")
    sentence = rule.sub('',raw_sentence)
    return sentence


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
            transforms.ToTensor(),
            ])  
    def extract(self, img):
        img = self.transform(img).unsqueeze(0)
        feature = self.model.forward(img).reshape(-1).squeeze()
        return feature.cpu().detach().numpy()


if __name__ == "__main__":
    with open('data/Gossiping-QA-Dataset.txt') as f:
        lines = f.readlines()
    chinese_corpus = []
    for line in lines[:100]:
        line = line.split()
        for i in line:
            if len(i) > 5:
                sentence = pruning(i)
                chinese_corpus.append(sentence[:10])
    cnn = CNN()
    file_folder = 'data/001'
    font = ImageFont.truetype("fonts/cwTeXYen-zhonly.ttf", 30)
    diffs = []
    raws = []

    for i in range(100):
        file_number = str(random.randint(1, 100))
        img_path = (file_folder + '/' + file_number + '.jpg')
        img = Image.open(img_path)
        width, height = img.size 

        raw_features = cnn.extract(img)
        raws.append(LA.norm(raw_features))

        image_editable = ImageDraw.Draw(img)
        wp = random.random() 
        wp = 0.1
        hp = 0.8
        image_editable.text((width*wp, height*hp), chinese_corpus[i], "black", font=font, spacing=20, align="center")
        
        text_features = cnn.extract(img)
        diff = LA.norm(raw_features - text_features)
        diffs.append(diff)

    print(np.mean(diffs), np.std(diffs))
    print(np.mean(raws), np.std(raws))


