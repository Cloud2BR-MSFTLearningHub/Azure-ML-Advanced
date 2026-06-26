---
title: 06 Training and AutoML
layout: default
nav_order: 8
---

# Training and AutoML

These diagrams show how AutoML explores model pipelines, what to expect from a run, and
how forecasting experiments expand the pipeline with time-aware feature engineering.

![AutoML diagram](https://raw.githubusercontent.com/brown9804/ML_DS_path/main/_docs/img/automl_diagram.png)

![AutoML process expectations](https://raw.githubusercontent.com/brown9804/ML_DS_path/main/_docs/img/automl_process_what_to_expect.png)

![Detailed ML-based time-series forecast steps](https://raw.githubusercontent.com/brown9804/ML_DS_path/main/_docs/img/Detailed-steps-of-ML-based-time-series-forecast.png)

AutoML workflow:

1. Choose task type
2. Provide training data
3. Select compute target
4. Set metric and constraints
5. Submit run and compare candidates

Performance relation:

$$
\text{Performance} = \frac{1}{\text{Execution Time}}
$$
