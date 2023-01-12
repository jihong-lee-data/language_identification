from datasets import load_dataset
import pandas as pd
import os

data_parquet_dir = "data/open_subtitles/"

langs = ['af', 'ar', 'bg', 'bn', 'br', 'bs', 'ca', 'cs', 'da', 'de', 'el', 'en', 'eo', 'es', 'et', 'eu', 'fa', 'fi', 'fr', 'gl', 'he', 'hi', 'hr', 'hu', 'hy', 'id', 'is', 'it', 'ja', 'ka', 'kk', 'ko', 'lt', 'lv', 'mk', 'ml', 'ms', 'nl', 'no', 'pl', 'pt', 'pt_br', 'ro', 'ru', 'si', 'sk', 'sl', 'sq', 'sr', 'sv', 'ta', 'te', 'th', 'tl', 'tr', 'uk', 'ur', 'vi', 'zh_cn', 'zh_tw']
langs.remove('en')
len(langs)

for l in langs:
    data_parquet_path = os.path.join(data_parquet_dir, f"{l}.parquet")
    print(data_parquet_path)
    try:
        tmp = load_dataset("open_subtitles", lang1="en", lang2=l, split= 'train')
        pd.DataFrame(tmp['translation']).to_parquet(data_parquet_path, compression = 'gzip')
    except:
        tmp = load_dataset("open_subtitles", lang1=l, lang2= "en", split= 'train')
        pd.DataFrame(tmp['translation']).to_parquet(data_parquet_path, compression = 'gzip')