import pandas as pd
import os
from tqdm import tqdm
from glob import glob

data_dir = "../data"
data_parquet_dir = "data/open_subtitles/"
# data_parquet_paths = glob(os.path.join("data/open_subtitles", "*.parquet"))

target_langs = ['en', 'es', 'pt_br', 'ro', 'tr', 'hu', 'sr', 'cs', 'pl', 'fr', 'el', 'bg', 'nl', 'it', 'hr', 'pt', 'he', 'ar', 'fi', 'ru', 'de', 'sl', 'sv', 'da', 'bs', 'et', 'zh_cn', 'id', 'sk', 'no', 'fa', 'zh_tw', 'vi', 'mk', 'th', 'ja', 'ms', 'sq', 'is', 'lt', 'ko', 'uk', 'eu', 'si', 'lv', 'ca', 'bn', 'ml', 'gl', 'ka', 'hi']
target_langs.remove('en')

n_each = 90000

agg_data_tsv_path = os.path.join(data_dir, f"os_data_{len(target_langs)}.tsv")




if agg_data_tsv_path in glob(os.path.join(data_dir, "*.tsv")):
    df_agg = pd.read_csv(agg_data_tsv_path)
else:
    df_agg = pd.DataFrame([], columns = ['labels', 'text'])

for lang in tqdm(target_langs):
    if lang in df_agg['labels']:
        print('already in aggregated data')
        continue
    tmp_raw = pd.read_parquet(os.path.join(data_parquet_dir, f"{lang}.parquet"))
    print(f"{lang}: {tmp_raw.shape[0]}")
    tmp_samples = tmp_raw.sample(n_each, random_state=42)
    tmp_melted = tmp_samples.melt(var_name = 'labels', value_name = 'text')
    df_agg = pd.concat([df_agg, tmp_melted])


# save data
pd.concat([df_agg.loc[df_agg['labels']!='en'], df_agg.loc[df_agg['labels']=='en'].sample(n_each, random_state =42)]).reset_index(drop= True).to_csv(agg_data_tsv_path, index = False)