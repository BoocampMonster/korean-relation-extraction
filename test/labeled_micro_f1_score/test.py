import pandas as pd
from sklearn import metrics
import argparse
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('--label-to-num', type=str, default='data/dict_label_to_num.pkl')
parser.add_argument('--validation', type=str, default='data/dev_data.csv')
parser.add_argument('--submission', type=str, default='save/Baseline/submission.csv')

args = parser.parse_args()

with open(args.label_to_num, 'rb') as f:
        dict_label_to_num = pickle.load(f)

validation = pd.read_csv(args.validation)
submission = pd.read_csv(args.submission)

true = validation.label.apply(lambda x: dict_label_to_num[x])
pred = submission.pred_label.apply(lambda x: dict_label_to_num[x])

labels = set(range(30))
assert(set(true.unique()) == labels)
assert(set(pred.unique()) == labels)

labels = list(range(1,30))
print(metrics.f1_score(true, pred, average='micro', labels=labels) * 100.0)
