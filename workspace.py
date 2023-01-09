import pandas as pd
from googletrans import Translator

raw = pd.read_excel('lang_detection_samples.xlsx')
print(raw.head(10))

translator = Translator()
result = translator.detect('안녕하세요')

print(result.lang, result.confidence)