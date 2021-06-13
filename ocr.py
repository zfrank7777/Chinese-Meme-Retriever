import easyocr
import Levenshtein as lev
import random
import urllib
from PIL import Image, ImageFont, ImageDraw
from WER import cal_wer
import re
import numpy as np
from tqdm import tqdm 
import os 
import cv2
import csv

## Generate random chinese words
def normal():
    head = random.randint(0xb0, 0xf7)
    body = random.randint(0xa1, 0xfe)
    val = f'{head:x}{body:x}'
    str = bytes.fromhex(val).decode('gb2312')
    return str

## Remove special characters
def pruning(raw_sentence):
    #sentence = re.sub('[A-Za-z0-9]+', '', raw_sentence)
    rule = re.compile(u"[^\u4e00-\u9fa5]")
    sentence = rule.sub('',raw_sentence)
    return sentence

## OCR model performance testing
def model_precision_test():
    with open('data/Gossiping-QA-Dataset.txt') as f:
        lines = f.readlines()

    chinese_corpus = []

    for line in lines:
        line = line.split()
        for i in line:
            if len(i) > 5:
                sentence = pruning(i)
                chinese_corpus.append(sentence)
    # print(chinese_corpus[:100])
    # print(len(chinese_corpus))
    # font = ImageFont.truetype("fonts/NotoSerifCJKtc-Bold.otf", 30)

    font = ImageFont.truetype("fonts/cwTeXYen-zhonly.ttf", 30)
    overall_error = []
    N = 10
    E_sum = 0
    C_sum = 0
    err_sum = 0
    R = []
    trange = tqdm(range(N), total=N)
    for i in trange:

        ground_truth = chinese_corpus[i][:10]

        file_folder = 1
        file_number = random.randint(0, 200)
        os.makedirs('test', exist_ok=True)
        out_path = 'test/{}-{}-{}.jpg'.format(i, file_folder, file_number)

        file_folder = 'data/{0:03d}'.format(file_folder)
        file_number = str(file_number)

        image_path = (file_folder + '/' + file_number + '.jpg')

        image = Image.open(image_path)
        width, height = image.size

        image_editable = ImageDraw.Draw(image)
        wp = random.random() 
        wp = 0.1
        hp = 0.8
        image_editable.text((width*wp, height*hp), ground_truth, "black", font=font, spacing=20, align="center")
        image.save(out_path)

        reader = easyocr.Reader(['ch_tra', 'en'])
        results = reader.readtext(out_path)
        text = ""
        score = 0
        ## TODO: Sometimes the chinese word does not have the highest confident score
        for result in results:
            confident_score = result[2]
            if confident_score > score:
                text = result[1]
                score = confident_score
        error = cal_wer(ground_truth, text)
        # print(error, lev.ratio(ground_truth, text))
        overall_error.append(error)

        R.append([out_path, ground_truth, text, error, len(ground_truth)])
        err_sum += error
        C_sum += len(ground_truth)
        E_sum += error * len(ground_truth)
        trange.set_postfix({'avg err': err_sum/(i+1), 'cer': E_sum/C_sum})
        

    print("overall accuracy = ", np.mean(overall_error))
    for r in R:
        print(r)

def generate_meme_context_file():
    output_path = "meme/context.csv"

    image = []
    #count = 0
    for files in os.listdir("meme/001"):
        if os.path.isfile("meme/001/"+files):
            #os.rename("meme/001/"+files,'meme/001/{0:04d}.jpg'.format(count))
            #count += 1
            image.append("meme/001/"+files)

    subtitles = []
    for i in tqdm(range(100)):

        image_path = image[i]
        pic_name = image_path.replace("meme/001/","")

        pic = cv2.imread(image_path)
        # cv2.imshow('test', pic)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        reader = easyocr.Reader(['ch_tra', 'en'])
        results = reader.readtext(pic)
        word = [pic_name]
        for result in results:
            if result[2] > 0.05:
                word.append(result[1])
        subtitles.append(word)  

    with open(output_path, 'w', encoding='utf-8-sig') as f:
        writer = csv.writer(f, delimiter=',')
        for subtitle in subtitles:
            writer.writerow(subtitle)


if __name__ == "__main__":
    generate_meme_context_file()
