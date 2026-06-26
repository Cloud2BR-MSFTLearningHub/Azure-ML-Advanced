---
title: 02 Azure ML Environment
layout: default
nav_order: 4
---

# Azure ML Environment

Main workspace assets:

- Workspace
- Compute Instance
- Compute Cluster
- Data assets
- Model registry
- Endpoints

## Workspace Taxonomy

![Azure ML workspace taxonomy](https://raw.githubusercontent.com/brown9804/ML_DS_path/main/_docs/img/azure-machine-learning-taxonomy.png)

![Azure ML environment taxonomy](https://raw.githubusercontent.com/brown9804/ML_DS_path/main/_docs/img/azure-ml-environment-taxonomy.png)

Key concepts:

- Experiment: a tracked training run.
- Registered model: trained artifact stored with version and lineage.
- Endpoint: deployment surface for scoring requests.

![Azure endpoint concept](https://raw.githubusercontent.com/brown9804/ML_DS_path/main/_docs/img/azure-endpoint-concept.png)

Compute guidance:

- Compute Instance for development
- Compute Cluster for scalable training
- ACI or AKS for serving

Practical split:

- AML Compute Cluster: training, sweeps, AutoML parallel iterations.
- AKS Inference Cluster: production-grade deployment and autoscaling.
