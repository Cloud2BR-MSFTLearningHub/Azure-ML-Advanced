---
title: 09 Deployment
layout: default
nav_order: 11
---

# Deployment

These visuals show the transition from training artifacts to production scoring
services, including the operational deployment flow.

![Training vs deployment model](https://raw.githubusercontent.com/brown9804/ML_DS_path/main/_docs/img/training_vs_deployment_model.png)

![ML deployment flow](https://raw.githubusercontent.com/brown9804/ML_DS_path/main/_docs/img/ml_deployment_flow.png)

![Deployment overview](https://raw.githubusercontent.com/brown9804/ML_DS_path/main/_docs/img/deployment_overview.png)

Deployment steps:

1. Register model
2. Build scoring script with init and run
3. Create inference environment
4. Validate local deployment
5. Deploy to ACI or AKS

```mermaid
flowchart LR
  A[Register] --> B[Score Script]
  B --> C[Inference Config]
  C --> D[Local Test]
  D --> E[ACI or AKS]
```
