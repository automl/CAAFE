### Usage
[![DEMO VIDEO](https://i.makeagif.com/media/5-20-2023/E4RfRM.gif)](https://www.youtube.com/watch?v=6zCD48d3kNU)

CAAFE lets you semi-automate your feature engineering process based on your explanations on the dataset and with the help of language models. It is based on the paper "LLMs for Semi-Automated Data Science: Introducing CAAFE for Context-Aware Automated Feature Engineering" by Hollmann, Müller, and Hutter (2023).
CAAFE systematically verifies the generated features to ensure that only features that are actually useful are added to the dataset.

To use CAAFE, first create a CAAFEClassifier object with the desired parameters:
```
caafe_clf = CAAFEClassifier(base_classifier=clf_no_feat_eng,
                      llm_model="gpt-4",
                      iterations=2)
```
Then, fit the classifier to your training data:
```
caafe_clf.fit_pandas(df_train,
               target_column_name=target_column_name,
               dataset_description=dataset_description,
              disable_caafe=False
              )
```
Finally, use the classifier to make predictions on your test data:
```
pred = caafe_clf.predict(df_test)
```

You can also try out the demo at: https://colab.research.google.com/drive/1mCA8xOAJZ4MaB_alZvyARTMjhl6RZf0a

For a minimal example of how to use CAAFE on your dataset, use CAFE_minimal.ipynb. To reproduce the experiments from the paper, use CAAFE.ipynb.


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
