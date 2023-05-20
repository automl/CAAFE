### Usage
```
caafe_clf = CAAFEClassifier(base_classifier=clf_no_feat_eng,
                      llm_model="gpt-4",
                      iterations=2)

caafe_clf.fit_pandas(df_train,
               target_column_name=target_column_name,
               dataset_description=dataset_description,
              disable_caafe=False 
              )

pred = caafe_clf.predict(df_test)
```

Try out our demo at: https://colab.research.google.com/drive/1mCA8xOAJZ4MaB_alZvyARTMjhl6RZf0a

Use CAFE_minimal.ipynb for a minimal example of how to use CAAFE on your dataset.
Use CAAFE.ipynb to reproduce the experiments from the paper.


### Paper
Hollmann, N., Müller, S., & Hutter, F. (2023). LLMs for Semi-Automated Data Science: Introducing CAAFE for Context-Aware Automated Feature Engineering
https://arxiv.org/abs/2305.03403

### License
[![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]

This work is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg