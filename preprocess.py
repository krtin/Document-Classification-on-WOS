import pathlib
import pandas as pd
import re


# copied from original paper
def clean_str(string):
    """
    Tokenization/string cleaning for dataset
    Every dataset is lower cased except
    """
    string = re.sub(r"\\", "", string)
    string = re.sub(r"\'", "", string)
    string = re.sub(r"\"", "", string)
    return string.strip().lower()


# copied from original paper
def text_cleaner(text):
    """
    cleaning spaces, html tags, etc
    parameters: (string) text input to clean
    return: (string) clean_text 
    """
    text = text.replace(".", "")
    text = text.replace("[", " ")
    text = text.replace(",", " ")
    text = text.replace("]", " ")
    text = text.replace("(", " ")
    text = text.replace(")", " ")
    text = text.replace("\"", "")
    text = text.replace("-", "")
    text = text.replace("=", "")
    rules = [
        {r'>\s+': u'>'},  # remove spaces after a tag opens or closes
        {r'\s+': u' '},  # replace consecutive spaces
        {r'\s*<br\s*/?>\s*': u'\n'},  # newline after a <br>
        {r'</(div)\s*>\s*': u'\n'},  # newline after </p> and </div> and <h1/>...
        {r'</(p|h\d)\s*>\s*': u'\n\n'},  # newline after </p> and </div> and <h1/>...
        {r'<head>.*<\s*(/head|body)[^>]*>': u''},  # remove <head> to </head>
        {r'<a\s+href="([^"]+)"[^>]*>.*</a>': r'\1'},  # show links instead of texts
        {r'[ \t]*<[^<]*?/?>': u''},  # remove remaining tags
        {r'^\s+': u''}  # remove spaces at the beginning
    ]
    for rule in rules:
        for (k, v) in rule.items():
            regex = re.compile(k)
            text = regex.sub(v, text)
        text = text.rstrip()
        text = text.strip()
    clean_text = text.lower()
    return clean_text


def get_dataset(clean_text=False):
    dataset_dir = pathlib.Path("WOS_dataset")
    with open(dataset_dir.joinpath("WOS5736", "X.txt"), "r") as f:
        X = f.readlines()
    with open(dataset_dir.joinpath("WOS5736", "Y.txt"), "r") as f:
        Y = f.readlines()
        
    with open(dataset_dir.joinpath("WOS5736", "YL1.txt"), "r") as f:
        YL1 = f.readlines()
    with open(dataset_dir.joinpath("WOS5736", "YL2.txt"), "r") as f:
        YL2 = f.readlines()

    # assert if equal
    assert len(X) == len(Y) == len(YL1) == len(YL2)

    dataset = []
    for x, y, yl1, yl2 in zip(X, Y, YL1, YL2):
        dataset.append((x.strip(), int(y.strip()), int(yl1.strip()), int(yl2.strip())))
    dataset = pd.DataFrame(dataset, columns=["X", "Y", "YL1", "YL2"])
    metadata = pd.read_excel(dataset_dir.joinpath("Meta-data", "Data.xlsx"))
    
    # Do a merge on X and abstracts left join
    dataset = dataset.merge(metadata.rename({"Y": "YL"}, axis=1), left_on='X', right_on='Abstract', how='left')

    # ensure that merge happened correctly with no null values
    assert dataset.isna().sum().sum() == 0

    if clean_text:
        dataset['X'] = dataset['X'].apply(text_cleaner)    

    # create 80/20 split
    train = dataset.sample(frac=0.8, random_state=42)
    test = dataset.drop(train.index)
    
    return train, test