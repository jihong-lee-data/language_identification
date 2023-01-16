import os
from module.engine import *
from glob import glob
import seaborn as sns
import matplotlib.pyplot as plt     
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix


# load model object
model_dir = "model/obj"

model_list = [p.split('.')[0].split('/')[-1] for p in glob(os.path.join(model_dir, "*pkl"))]


model_name = input(f'''Please input model name \n (model list: {model_list}): \n''')
# model_name = 'mnnb_os_data_51'

model_path = os.path.join(model_dir, f"{model_name}.pkl")

model = load_model(model_path)


# load dataset
dataset_name = 'os_data_51'
dataset_dir = os.path.join('data', dataset_name)
dataset = load_from_disk(dataset_dir)['test']

results_name = f'{model_name}_{dataset_name}'

# feature preprocessing
x = preprocessor(dataset['text'])
y = dataset.features['labels'].int2str(dataset['labels'])



# calc model score & save result
results = dict()
y_pred = model.predict(x)

# confusion matrix
cm = confusion_matrix(y, y_pred, labels = dataset.features['labels'].names)

plt.figure(figsize = (35, 30))
ax= plt.subplot()
sns.heatmap(cm, annot=True, fmt='g', ax=ax, cmap = "OrRd", cbar = False)  #annot=True to annotate cells, ftm='g' to disable scientific notation

# labels, title and ticks
ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels')
ax.set_title('Confusion Matrix')
ax.xaxis.set_ticklabels(dataset.features['labels'].names)
ax.yaxis.set_ticklabels(dataset.features['labels'].names)

plt.savefig(f'cm_{results_name}.png')

df_results = pd.DataFrame(np.column_stack([dataset['text'], y, y_pred]), columns = ['text', 'label_true', 'label_pred'])
df_results.to_csv(f"{results_name}_inference.csv", index = False)

df_ic = df_results.loc[df_results['label_true'] != df_results['label_pred']]

df_ic.to_csv(f"{results_name}_incorrect.csv", index = False)