import pandas as pd
import glob
import re

arglist = ['Job', 'Epochs', 'Dropout', 'Hidden', 'Embed']

records = []

for file in glob.glob("./*.log"):
    if file.startswith('./output'):
        scores = [line for line in open(file) if re.match(
            r'Your score on validation data: (\d*.\d*)', line)]
        scores = [re.findall(r'Your score on validation data: (\d*.\d*)', line)
                  for line in scores]
        score = float("".join(scores[0]))
        args = re.split(r'-', file)[1:]  # remove filename
        args[-1] = args[-1][:-4]  # remove '.log'
        args = dict(zip(arglist, args))
        args['Score'] = score
        args['Batch Size'] = 32
        records.append(args)

df = pd.DataFrame.from_dict(records)
df = df[['Job', 'Epochs', 'Batch Size', 'Embed', 'Hidden', 'Dropout', 'Score']]
print(df)
df.to_csv('./record.csv', index=False)
