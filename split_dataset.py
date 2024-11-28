import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split



def assign_tag(df):
    # roughl_threshold = 0.7
    nli_threshold = 0.9
    reverse_nli_threshold = 0.9
    if df['nli'] >= nli_threshold and df['reverse_nli'] >= reverse_nli_threshold:
        return 1
    elif df['nli'] < nli_threshold or df['reverse_nli'] < reverse_nli_threshold:
        return 0
    # elif df['rougeL'] < roughl_threshold:
    #     return 0


path = "D:\\SunYihao\\eecs498\\impossible-distillation-rep\\gen_data\\stage0\\gpt2-xl"

jsonl_files = [f for f in os.listdir(path) if f.endswith('.jsonl')]

dataframes = []

# Read each .jsonl file
for file in jsonl_files:
    file_path = os.path.join(path, file)
    with open(file_path, 'r', encoding='utf-8') as f:
        # Load JSON lines
        data = [json.loads(line) for line in f]
        # Convert to DataFrame
        df = pd.DataFrame(data)
        dataframes.append(df)

# Combine all DataFrames
combined_df = pd.concat(dataframes, ignore_index=True)

# Shuffle the data to ensure randomness
combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
combined_df['tag'] = combined_df.apply(assign_tag, axis=1)

cnt_1 = (combined_df['tag'] == 1).sum()
cnt_0 = (combined_df['tag'] == 0).sum()

print(cnt_1, "\t", cnt_0, "\t", cnt_1/cnt_0)

train_ratio = 5 / 7
dev_test_ratio = 1 / 7

# Split the dataset
train_df, dev_test_df = train_test_split(combined_df, test_size=(1 - train_ratio), random_state=42)
dev_df, test_df = train_test_split(dev_test_df, test_size=0.5, random_state=42)

train_df = train_df.reset_index(drop=True)
train_df['id'] = train_df.index
train_df = train_df[['id'] + [col for col in train_df.columns if col != 'id']]

dev_df = dev_df.reset_index(drop=True)
dev_df['id'] = dev_df.index
dev_df = dev_df[['id'] + [col for col in dev_df.columns if col != 'id']]

test_df = test_df.reset_index(drop=True)
test_df['id'] = test_df.index
test_df = test_df[['id'] + [col for col in test_df.columns if col != 'id']]

train_path = os.path.join(path, 'train.tsv')
dev_path = os.path.join(path, 'dev.tsv')
test_path = os.path.join(path, 'test.tsv')

train_df.to_csv(train_path, sep='\t', index=False)
dev_df.to_csv(dev_path, sep='\t', index=False)
test_df.to_csv(test_path, sep='\t', index=False)

print(f"Train, dev, and test splits saved at: {path}")