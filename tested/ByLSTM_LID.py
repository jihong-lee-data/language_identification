import pandas as pd
from LanguageIdentifier import predict, rank
from tqdm import tqdm


raw = pd.read_excel('lang_detection_samples.xlsx')
print('data loaded')

print('detecting languages...')

results = []

for s in tqdm(raw['inputText'].astype(str)):
    results.append({'lang': predict(s), 'rank': rank(s)})

raw.join(pd.DataFrame(results)).to_csv("LSTM-LID_results.csv", index = False)

print('results saved')