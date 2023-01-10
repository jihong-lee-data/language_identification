from datasets import load_dataset, load_from_disk
import os

def load_hf_dataset(dataset_path:str, save_to_disk: bool = False):
    if not os.path.isdir(dataset_path):
        print('Downloading dataset...')
        dataset = load_dataset(dataset_path)
        if save_to_disk:
            dataset.save_to_disk(dataset_path)
        print('The dataset is saved and loaded.')
    else: 
        print('The dataset already exists.')
        dataset = load_from_disk(dataset_path)
        print('The dataset is loaded.')
    return dataset







