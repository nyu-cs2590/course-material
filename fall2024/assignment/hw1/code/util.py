import json
import math
import nltk

########### Text classification ###########

def read_dataset(path, max_num_examples=-1):
    with open(path, 'r') as fin:
        data = json.load(fin)
        print('Load {} examples from {}'.format(len(data), path))
    return data[:max_num_examples] if max_num_examples > 0 else data

def dot(v1, v2):
    """Return the dot product of two vectors.
    Parameters:
        v1 : dict
            feature (str) to value (float)
        v2 : same type as v1
    Returns:
        float
    """
    if len(v1) < len(v2):
        return dot(v2, v1)
    else:
        return sum([v1.get(f, 0) * v for f, v in v2.items()])

def increment(v1, v2, scale):
    """Update v1 to v1 + scale * v2.
    Note that v1 is updated in place.
    Parameters:
        v1 : dict
            feature (str) to value (float)
        v2 : same type as v1
        scale : float
    Returns:
        None
    """
    for f, v in v2.items():
        v1[f] = v1.get(f, 0) + scale * v

def predict(weights, feat):
    """Return p(y=1|x) using the logistic regression model.
    """
    return 1. / (1 + math.exp(-dot(weights, feat)))

def verbose_predict(feat, weights, label, fout):
    """Print details of prediction.
    """
    y_hat = 1 if dot(weights, feat) > 0 else 0
    correct = y_hat == label
    fout.write('label={}, predicted={}, {}\n'.format(label, y_hat, 'correct' if correct else 'wrong'))
    for f, v in sorted(list(feat.items()), key=lambda x: x[1] * weights.get(x[0], 0), reverse=True):
        w = weights.get(f, 0)
        fout.write('f={}, v={}, w={}, v*w={}\n'.format(f, v, w, v * w))
    return y_hat

def error_analysis(examples, feature_extractor, weights, out):
    """Output detailed prediction information to file.
    """
    with open(out, 'w') as fout:
        for ex in examples:
            fout.write('='*80 + '\n')
            fout.write('P: {}\nH: {}\n'.format(' '.join(ex['sentence1']),
                ' '.join(ex['sentence2'])))
            feat = feature_extractor(ex)
            verbose_predict(feat, weights, ex['gold_label'], fout)

def evaluate_predictor(examples, predictor):
    """Return the average 0-1 loss, i.e. the error rate.
    Parameters:
        examples : [tuple]
            [(feature vector (dict), label (int={0,1}))]
        predictor : function
            example --> predicted label (int={0,1})
    Returns:
        average 0-1 loss (float)
    """
    return sum([predictor(feat) != label for feat, label in examples]) / float(len(examples))


########### Word embedding  ###########

def read_corpus():
    nltk.download('gutenberg')
    from nltk.corpus import gutenberg
    tokens = [w.lower() for w in list(gutenberg.words('austen-emma.txt'))]
    return tokens
