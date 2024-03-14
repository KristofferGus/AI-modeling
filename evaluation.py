
from nltk.translate.meteor_score import meteor_score
from nltk import word_tokenize
from tqdm import tqdm
import matplotlib.pyplot as plt



def write_references_to_file(references, reference_file):
    open(reference_file, 'w').close()
    with open(reference_file, "a") as f:
        for reference in references:
            f.write(reference + "\n")



def write_translatons_to_file(input_file, prediction_file, translator_func):
    with open(input_file, "r") as f:
        open(prediction_file, 'w').close()
        with open(prediction_file, "a") as g:
            for line in tqdm(f.readlines()):
                line = line.removesuffix("\n")
                prediction = translator_func(line)
                g.write(prediction + "\n")

def evaluate_score(reference, prediction):
    return meteor_score([word_tokenize(reference)], word_tokenize(prediction), alpha = 0.9, beta = 3, gamma = 0.5)

def evaluate_model(reference_file, prediction_file):
    xs = []
    with open(reference_file, "r") as f:
        with open(prediction_file, "r") as g:
            for ref, pred in zip(f.readlines(), g.readlines()):
                xs.append(evaluate_score(ref, pred))
    return sum(xs) / len(xs)
