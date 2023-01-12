import pandas as pd
import os
from tqdm import tqdm

langs = ['af', 'ar', 'bg', 'bn', 'br', 'bs', 'ca', 'cs', 'da', 'de', 'el', 'en', 'eo', 'es', 'et', 'eu', 'fa', 'fi', 'fr', 'gl', 'he', 'hi', 'hr', 'hu', 'hy', 'id', 'is', 'it', 'ja', 'ka', 'kk', 'ko', 'lt', 'lv', 'mk', 'ml', 'ms', 'nl', 'no', 'pl', 'pt', 'pt_br', 'ro', 'ru', 'si', 'sk', 'sl', 'sq', 'sr', 'sv', 'ta', 'te', 'th', 'tl', 'tr', 'uk', 'ur', 'vi', 'zh_cn', 'zh_tw']
langs.remove('en')
data_parquet_dir = "data/open_subtitles/"
n_each = 15000

df_agg = pd.DataFrame([], columns = ['label', 'text'])

for l in tqdm(langs[:7]):
    data_parquet_path = os.path.join(data_parquet_dir, f"{l}.parquet")
    tmp_raw = pd.read_parquet(data_parquet_path)
    print(f"{l}: {tmp_raw.shape[0]}")
    tmp_samples = tmp_raw.sample(n_each, random_state=42)
    tmp_melted = tmp_samples.melt(var_name = 'label', value_name = 'text')
    df_agg = pd.concat([df_agg, tmp_melted])