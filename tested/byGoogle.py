import pandas as pd
from googletrans import Translator
from tqdm import tqdm


raw = pd.read_excel('lang_detection_samples.xlsx')
print('data loaded')


translator = Translator()

print('detecting languages...')


results_lst = []

for s in tqdm(raw['inputText']):
    tmp = translator.detect(s)
    results_lst.append([tmp.lang, tmp.confidence])

df_results = pd.DataFrame(results_lst, columns = ['lang', 'confidence'])

raw.join(df_results).to_csv("googletrans_results.csv", index = False)

print('results saved')