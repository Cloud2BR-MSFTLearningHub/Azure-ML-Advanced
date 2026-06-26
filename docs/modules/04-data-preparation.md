---
title: 04 Data Preparation
layout: default
nav_order: 6
---

# Data Preparation

This sequence illustrates the data lifecycle: from business framing, to collecting and
engineering features, to defining enough volume for model reliability.

![ML process by stages](https://raw.githubusercontent.com/brown9804/ML_DS_path/main/_docs/img/ml_process_by_stages.png)

![Collect data and targets](https://raw.githubusercontent.com/brown9804/ML_DS_path/main/_docs/img/collect_data_init_primary_second_targets.png)

![Feature engineering while collecting data](https://raw.githubusercontent.com/brown9804/ML_DS_path/main/_docs/img/feature_engineering_collect_data.png)

![Dataset size examples by project](https://raw.githubusercontent.com/brown9804/ML_DS_path/main/_docs/img/examples_of_datasets_for_diff_projects.png)

Checklist:

- Remove duplicates and nulls
- Validate schema and dtypes
- Split train and test sets
- Register datasets in Azure ML

Useful split:

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
```

The next visuals reinforce how supervised datasets are split and validated before
training, plus a dtype reference to prevent schema and conversion errors.

![Training/testing data flow](https://raw.githubusercontent.com/brown9804/ML_DS_path/main/_docs/img/training_testing_data_flow.png)

![Training and test split](https://raw.githubusercontent.com/brown9804/ML_DS_path/main/_docs/img/training_test_split.png)

![Python dtype overview](https://raw.githubusercontent.com/brown9804/ML_DS_path/main/_docs/img/python_dtype.png)
