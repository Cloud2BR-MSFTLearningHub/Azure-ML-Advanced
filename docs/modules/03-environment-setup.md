---
title: 03 Environment Setup
layout: default
nav_order: 5
---

# Environment Setup

The two diagrams below show how Azure ML assets are organized at workspace level and how
environments are reused across training and inference.

![Azure ML workspace taxonomy](https://raw.githubusercontent.com/brown9804/ML_DS_path/main/_docs/img/azure-machine-learning-taxonomy.png)

![Azure ML environment taxonomy](https://raw.githubusercontent.com/brown9804/ML_DS_path/main/_docs/img/azure-ml-environment-taxonomy.png)

Typical setup:

```console
conda env create --name aml-env --file ./dependencies/environment.yml --force
conda activate aml-env
pip install -r ./dependencies/requirements.txt
```

Validation:

```console
pip show scikit-learn
pip show azureml-sdk
conda env list
```

These references help when sizing compute and understanding memory/number representation
concepts that affect performance decisions.

![Binary vs decimal data measurements](https://raw.githubusercontent.com/brown9804/ML_DS_path/main/_docs/img/binary_vs_decimal_data_measurements.png)

![Summary of number systems](https://raw.githubusercontent.com/brown9804/ML_DS_path/main/_docs/img/summary_of_number_system.png)
