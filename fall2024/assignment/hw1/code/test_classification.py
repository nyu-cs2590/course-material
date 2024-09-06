from submission import extract_unigram_features, extract_custom_features, learn_predictor
from util import *

def test_unigram():
    train_data = read_dataset('data/train.json', -1)
    valid_data = read_dataset('data/dev.json', -1)
    feature_extractor = extract_unigram_features
    weights = learn_predictor(train_data, valid_data, feature_extractor, 0.01, 10)
    predictor = lambda ex: 1 if dot(weights, feature_extractor(ex)) > 0 else 0
    train_err = evaluate_predictor([(ex, ex['gold_label']) for ex in train_data], predictor)
    valid_err = evaluate_predictor([(ex, ex['gold_label']) for ex in valid_data], predictor)
    print('train error={}, valid error={}'.format(train_err, valid_err))
    error_analysis(valid_data[:100], feature_extractor, weights, 'error_analysis_unigram.txt')

def test_custom():
    train_data = read_dataset('data/train.json', -1)
    valid_data = read_dataset('data/dev.json', -1)
    feature_extractor = extract_custom_features
    weights = learn_predictor(train_data, valid_data, feature_extractor, 0.01, 10)
    predictor = lambda ex: 1 if dot(weights, feature_extractor(ex)) > 0 else 0
    train_err = evaluate_predictor([(ex, ex['gold_label']) for ex in train_data], predictor)
    valid_err = evaluate_predictor([(ex, ex['gold_label']) for ex in valid_data], predictor)
    print('train error={}, valid error={}'.format(train_err, valid_err))
    error_analysis(valid_data[:100], feature_extractor, weights, 'error_analysis_custom.txt')

test_unigram()
test_custom()


