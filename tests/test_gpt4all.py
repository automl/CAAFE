from caafe import CAAFEClassifier # Automated Feature Engineering for tabular datasets
from tabpfn import TabPFNClassifier # Fast Automated Machine Learning method for small tabular datasets
from sklearn.ensemble import RandomForestClassifier

import os
import torch
from caafe import data
from sklearn.metrics import accuracy_score, roc_auc_score
from tabpfn.scripts import tabular_metrics
from functools import partial

metric_used = tabular_metrics.auc_metric
cc_test_datasets_multiclass = data.load_all_data()

ds = cc_test_datasets_multiclass[5]
ds, df_train, df_test, _, _ = data.get_data_split(ds, seed=0)
target_column_name = ds[4][-1]
dataset_description = ds[-1]
ds[0]

from caafe.preprocessing import make_datasets_numeric
df_train, df_test = make_datasets_numeric(df_train, df_test, target_column_name)
train_x, train_y = data.get_X_y(df_train, target_column_name)
test_x, test_y = data.get_X_y(df_test, target_column_name)

### Setup Base Classifier

# clf_no_feat_eng = RandomForestClassifier()
clf_no_feat_eng = TabPFNClassifier(device=('cuda' if torch.cuda.is_available() else 'cpu'), N_ensemble_configurations=4)
clf_no_feat_eng.fit = partial(clf_no_feat_eng.fit, overwrite_warning=True)

clf_no_feat_eng.fit(train_x, train_y)
pred = clf_no_feat_eng.predict(test_x)
acc = accuracy_score(pred, test_y)
print(f'Accuracy BEFORE CAAFE {acc}')
#roc_auc = roc_auc_score(pred, test_y)
#print(f'ROC auc BEFORE CAAFE {roc_auc}')

### Setup and Run CAAFE - This will be billed to your OpenAI Account (in case you use it with llm_model with any openai model)!

## OBS: You need to manually download the model files
models_list = ["ggml-model-gpt4all-falcon-q4_0.bin",
               "starcoderbase-3b-ggml.bin",
               "starcoderbase-7b-ggml.bin",
               "llama-2-7b-chat.ggmlv3.q4_0.bin",
               "nous-hermes-13b.ggmlv3.q4_0.bin",
               "wizardlm-13b-v1.1-superhot-8k.ggmlv3.q4_0.bin"]

os.environ["GPT4ALL_MODEL_BIN"] = f"<FULL PATH TO MODELS DIRECTORY>/{models_list[1]}"

caafe_clf = CAAFEClassifier(base_classifier=clf_no_feat_eng,
                            iterations=10,
                            llm_model="gpt4all",
                            display_method="print")

caafe_clf.fit_pandas(df_train,
                     target_column_name=target_column_name,
                     dataset_description=dataset_description)

pred = caafe_clf.predict(df_test)
acc = accuracy_score(pred, test_y)
print(f'Accuracy AFTER CAAFE {acc}')
#roc_auc = roc_auc_score(pred, test_y)
#print(f'ROC auc AFTER CAAFE {roc_auc}')


print(caafe_clf.code)