---
title: 00 Introduction and Lifecycle
layout: default
nav_order: 2
---

# Introduction and ML Lifecycle

Azure Machine Learning organizes the end-to-end lifecycle:

1. Problem framing
2. Data preparation
3. Training and validation
4. Model registration
5. Deployment
6. Monitoring and retraining

## Legacy Visuals (from original 04k notes)

![ML infrastructure tools for production](https://raw.githubusercontent.com/brown9804/ML_DS_path/main/_docs/img/ml-infrastructure-tools-for-production.png)

![ML workflow stages](https://raw.githubusercontent.com/brown9804/ML_DS_path/main/_docs/img/ml_workflow_stages.png)

![Overview ML flow](https://raw.githubusercontent.com/brown9804/ML_DS_path/main/_docs/img/Overview_ML_flow.png)

## Web Service vs API

- A deployed Azure ML model is typically exposed as a REST API endpoint.
- In practice, teams often say "web service" for the deployed scoring interface.

![Web service vs API](https://raw.githubusercontent.com/brown9804/ML_DS_path/main/_docs/img/webservice_vs_api.png)

![Web service vs API table](https://raw.githubusercontent.com/brown9804/ML_DS_path/main/_docs/img/table_webservice_vs_api.png)

```mermaid
flowchart LR
  A[Problem] --> B[Data]
  B --> C[Train]
  C --> D[Register]
  D --> E[Deploy]
  E --> F[Monitor]
  F --> C
```
