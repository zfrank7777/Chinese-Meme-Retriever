import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import csv
from PIL import Image

from tqdm import tqdm


def read_files():
    images = {}
    img2id = {}
    id2img = []

    fn2dir = {
        'context.csv': '001',
        'context2.csv': '2'
    }

    # Read Files
    images_with_text = {}
    for fileName, dirName in fn2dir.items():
        with open(os.path.join('meme', fileName), 'r', newline='') as f:
            reader = csv.reader(f)
            for line in reader:
                img = os.path.join('meme', dirName, line[0]).replace('\ufeff', '')
                images_with_text[img] = {'text': ''.join(line[1:]), 'path': img}
    keys1 = set(images_with_text.keys())
    
    #  Read CNN features
    with open('meme/cnn_features.pkl', 'rb') as f:
        features = pickle.load(f)
    keys2 = set(features.keys())

    id2img = list(keys1 & keys2)
    print('drop', (keys1 | keys2) - set(id2img))
    img2id = {e: i for i, e in enumerate(id2img)}
    images = {i: images_with_text[e] for i, e in enumerate(id2img)}

    for i, k in enumerate(id2img):
        images[i]['cnn_features'] = features[k]
    cnn_features_mat = np.stack([features[k] for k in id2img])
    cnn_features_mat = cnn_features_mat / ((cnn_features_mat ** 2).mean(-1, keepdims=True) ** 0.5 + 1e-30)

    # Build Inverted File
    inverted_file = {}
    lens = []
    for img, item in tqdm(images.items(), total=len(images.keys())):
        grams = {}
        t = item['text']
        lens.append(len(t))
        for i, c in enumerate(t):
            grams[c] = grams.get(c, 0) + 1 
            if i > 0:
                bigram = t[i-1]+t[i]
                grams[bigram] = grams.get(bigram, 0) + 1 
        for g, n in grams.items():
            inverted_file[g] = inverted_file.get(g, {})
            inverted_file[g][img] = n

    return images, inverted_file, np.mean(lens), img2id, id2img, cnn_features_mat
    """
    # Read Inverted File cache if exists
    inverted_file_pkl = os.path.join('vsm/inverted-file.pkl')
    if os.path.exists(inverted_file_pkl):
        with open(inverted_file_pkl, 'rb') as f:
            inverted_file = pickle.load(f)
    else:
        # Build Inverted File
        inverted_file = {}
        for img, item in tqdm(images.items(), total=len(images.keys())):
            grams = {}
            t = item['text']
            for i, c in enumerate(t):
                grams[c] = grams.get(c, 0) + 1 
                if i > 0:
                    bigram = t[i-1]+t[i]
                    grams[bigram] = grams.get(bigram, 0) + 1 
            for g, n in grams.items():
                inverted_file[g] = inverted_file.get(g, {})
                inverted_file[g][img2id[img]] = n

        print("Dumping file: inverted-file.pkl  ...")
        with open(inverted_file_pkl, 'wb') as f:
            pickle.dump(inverted_file, f)
    """


def write_csv(results, filename):
    with open(filename, 'w') as f:
        f.write('query_id,retrieved_docs\n')
        for qid, doclist in results.items():
            doc_str = " ".join(doclist)
            ansline = '{},{}'.format(qid, doc_str)
            f.write(ansline+'\n')


def calc_AP(myans, gt):
    ap = 0
    hit = 0
    for i, doc in enumerate(myans):
        if doc in gt:
            hit += 1
            ap += hit/(i+1)
        if hit == len(gt):
            break
    ap /= hit
    return ap


def evaluate(results, answers):
    MAP = 0
    ans_len = 0
    for qid, result in results.items():
        assert qid in answers.keys()
        AP = calc_AP(result, answers[qid])
        MAP += AP
        ans_len += len(answers[qid])
    ans_len /= len(results)
    MAP /= len(results)
    print('MAP: ', MAP)


def show(images, size=500, row=5):
    merged = np.zeros((size*row, size*row, 3), dtype=np.uint8)
    for i, fn in enumerate(images[:row*row]):
        x, y = i % row * size, i // row * size;
        img = Image.open(fn)
        scale = min(size / img.width, size / img.height)
        dim = w, h = round(scale * img.width), round(scale * img.height)
        img = np.asarray(img.resize((w, h)).convert('RGB'))
        assert img.dtype == np.uint8
        before = [(size - d) // 2 for d in dim]
        after = [size - d - b for d, b in zip(dim, before)]
        img = np.pad(img, list(zip(before, after))[::-1] + [(0, 0)])
        assert img.shape == (size, size, 3)
        merged[y:y+size, x:x+size] = img
    plt.imshow(merged)
    plt.show()


if __name__ == '__main__':
    read_files()
