import argparse
import os
import logging
import math
import numpy as np

from utils import read_files, show


logging.basicConfig(level=logging.INFO)


def process_query(query, use_bi=True):
    terms = []
    for i, c in enumerate(query):
        terms.append(c)
        if use_bi:
            if i > 0:
                terms.append(query[i-1]+query[i])
    return terms


def standardize(x):
    return (x - x.mean()) / (x.std() + 1e-30)


def retrieve(query, images, inverted_file, avg_len):
    # Arguments
    k = 1.2
    b = 0.75

    query_terms = process_query(query)
    N = len(images.keys())
    score = {}
    for term in query_terms:
        if term not in inverted_file.keys():
            # logging.warning('term not in inverted file: {}'.format(term))
            continue
        df = len(inverted_file[term].keys())
        idf = math.log(((N-df+0.5)/(df+0.5))+1)
        for img, tf in inverted_file[term].items():
            D = len(images[img]['text'])
            tf = (k+1)*tf / (tf+(k*(1-b+b*D/avg_len)))
            score[img] = score.get(img, 0) + tf * idf

    top100 = sorted(score.keys(),
                    key=lambda x: score[x], reverse=True)[:100]
    return score, top100


def get_feedback_query(imgs, images, inverted_file,
                       k=1.2, top_term=10):

    N = len(images.keys())
    new_terms = []
    for img in imgs:
        text = images[img]['text']
        new_terms += process_query(text)
    new_terms = list(set(new_terms))

    score = {}
    for term in new_terms:
        if term not in inverted_file.keys():
            continue
        df = len(inverted_file[term].keys())
        idf = math.log(((N-df+0.5)/(df+0.5))+1)
        for doc, tf in inverted_file[term].items():
            tf = (k+1) * tf / (tf + k)
            score[term] = score.get(term, 0) + tf * idf
    feedback_terms = sorted(score.keys(),
                            key=lambda x: score[x], reverse=True)[:top_term]
    return feedback_terms


def main(args):
    logging.info('Reading files ...')
    images, inverted_file, avg_len, img2id, id2img, features = read_files()
    N = len(images.keys())

    while True:
        logging.info('Enter query:')
        query = input()

        q_score, top100 = retrieve(query, images, inverted_file, avg_len)
        filenames = [id2img[i] for i in top100]
        print('before feedback')
        for fn in filenames[:10]:
            print(fn, img2id[fn], images[img2id[fn]]['text'])
        q_score = np.array([q_score.get(i, 0) for i in range(N)])
        q_score = standardize(q_score)

        for _ in range(args.feedback_steps):
            # Visual feedback
            v_score = standardize(features @ features[top100[:args.num_visual_doc]].mean(0))
            v_score = standardize(v_score)
            # print(v_score)

            # Text feedback
            fb_query = get_feedback_query(top100[:args.num_feedback_doc],
                                          images, inverted_file,
                                          top_term=args.num_feedback_term)
            fb_score, _ = retrieve(fb_query, images, inverted_file, avg_len)
            fb_score = np.array([fb_score.get(i, 0) for i in range(N)])
            fb_score = standardize(fb_score)

            score = q_score + fb_score * args.feedback_weight + v_score * args.visual_weight

            top100 = score.argsort()[::-1][:100]

        filenames = [id2img[i] for i in top100]
        print('after feedback')
        for fn in filenames[:10]:
            print(fn, img2id[fn], images[img2id[fn]]['text'])
        show(filenames)


def _parse_args():
    parser = argparse.ArgumentParser(description='VSM model with RF')
    parser.add_argument('-r',
                        help='use relevant feedback',
                        action='store_true')
    parser.add_argument('-f', '--feedback_weight',
                        help='feedback score weight',
                        type=float,
                        default=0.5)
    parser.add_argument('-v', '--visual_weight',
                        help='visual feedback score weight',
                        type=float,
                        default=0.5)
    parser.add_argument('-s', '--feedback_steps',
                        help='feedback steps',
                        type=int,
                        default=4)
    parser.add_argument('-n', '--num_feedback_doc',
                        help='number of feedback documents',
                        type=int,
                        default=10)
    parser.add_argument('-nv', '--num_visual_doc',
                        help='number of visual feedback documents',
                        type=int,
                        default=10)
    parser.add_argument('-t', '--num_feedback_term',
                        help='number of feedback terms',
                        type=int,
                        default=20)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = _parse_args()
    main(args)
