import jiwer


def cal_wer(ground_truth, hypothesis):
    transformation = jiwer.Compose([
        jiwer.ToLowerCase(),
        jiwer.ExpandCommonEnglishContractions(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.RemoveWhiteSpace(replace_by_space=False),
    ])

    wer = jiwer.wer(
        ground_truth,
        hypothesis,
        truth_transform=transformation,
        hypothesis_transform=transformation
    )
    return wer


if __name__ == "__main__":

    gt = "因為屈原是台灣人"
    pred = "因為屈原是台人"
    wer = cal_wer(gt, pred)
    print(wer)
    """
    for dirPath, dirNames, fileNames in os.walk(wave2vec_root):
        wave2vec_files = fileNames
        break
    """
