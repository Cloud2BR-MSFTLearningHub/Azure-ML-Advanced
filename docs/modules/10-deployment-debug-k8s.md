---
title: 10 Deployment Debugging
layout: default
nav_order: 12
---

# Deployment Debugging with Kubernetes

These quality visuals are included as quick troubleshooting references when an endpoint
behaves unexpectedly after deployment.

![Confusion matrix quality reference](https://raw.githubusercontent.com/brown9804/ML_DS_path/main/_docs/img/confusion_matrix_good_bad.png)

![Lift curve quality reference](https://raw.githubusercontent.com/brown9804/ML_DS_path/main/_docs/img/lift_good_bad.png)

![ROC quality reference](https://raw.githubusercontent.com/brown9804/ML_DS_path/main/_docs/img/roc_good_bad.png)

Key tools:

- kubectl
- kind
- minikube
- kubeadm

Useful commands:

```bash
kubectl get pods
kubectl describe pod <pod-name>
kubectl logs <pod-name>
```
