import csv
import json
import random
import argparse
import re

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input')
    parser.add_argument('--output')
    parser.add_argument('--num-examples', default=10, type=int)
    args = parser.parse_args()
    return args

def main(args):
    examples = []
    with open(args.input, 'r') as fin, open(args.output, 'w') as fout:
        reader = csv.DictReader(fin, delimiter='\t')
        for row in reader:
            if not row['gold_label'] in ('neutral', 'entailment', 'contradition'):
                continue
            examples.append(row)
        examples = random.choices(examples, k=args.num_examples)
        examples = [{
            'gold_label': 1 if e['gold_label'] == 'entailment' else 0,
            'sentence1': re.sub(r'[() ]+', ' ', e['sentence1_binary_parse']).strip().split(),
            'sentence2': re.sub(r'[() ]+', ' ', e['sentence2_binary_parse']).strip().split(),
            }
            for e in examples]
        fout.write(json.dumps(examples, indent=2))


if __name__ == '__main__':
    random.seed(42)
    args = parse_args()
    main(args)


