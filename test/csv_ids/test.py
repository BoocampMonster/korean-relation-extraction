import argparse
import pandas as pd

parser = argparse.ArgumentParser()

parser.add_argument('-f', '--file',
                    type=str, default='~/dataset/test/test_data.csv',
                    help='csv file path')

args = parser.parse_args()
print(args)

df = pd.read_csv(args.file)

assert((df['id'].values == df.index.values).all())

print('Test passed.')
