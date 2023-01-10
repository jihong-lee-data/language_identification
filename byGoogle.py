import pandas as pd
from googletrans import Translator


raw = pd.read_excel('lang_detection_samples.xlsx')
print('data loaded')


translator = Translator()

print('detecting languages...')
raw_results = translator.detect(raw['inputText'].tolist())

df_results = pd.DataFrame([(r.lang, r.confidence) for r in raw_results], columns = ['lang', 'confidence'])

raw.join(df_results).to_csv("googletrans_results.csv", index = False)