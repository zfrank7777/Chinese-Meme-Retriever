import os
import pickle
import numpy as np
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
    i = 0
    for fileName, dirName in fn2dir.items():
        with open(os.path.join('meme', fileName), 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.replace('\n', '')
                img = os.path.join('meme', dirName, line.split(',')[0])
                img2id[img] = i
                images[i] = {'text': ''.join(line.split(',')[1:])}
                id2img.append(img)
                i += 1
    
    #  Read CNN features
    with open('meme/cnn_features.pkl', 'rb') as f:
        features = pickle.load(f)
    for k, v in features.items():
        if k in images.keys():
            images[k]['cnn_features'] = v

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

    return images, inverted_file, np.mean(lens), img2id, id2img
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


if __name__ == '__main__':
    read_files()
