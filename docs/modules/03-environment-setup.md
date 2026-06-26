---
title: 03 Environment Setup
layout: default
nav_order: 5
---

# Environment Setup

This module helps beginners build a reproducible runtime from zero. Reproducibility is
critical because model behavior depends on package versions, OS libraries, and Python
runtime details.

## Why environment reproducibility matters

- Same code can produce different results under different dependency versions.
- Training and inference must share compatible libraries.
- Teams need deterministic rebuilds for audits and incident recovery.

The diagrams below show how Azure ML assets are organized and how environments are reused
across training and inference.

![Azure ML workspace taxonomy](https://raw.githubusercontent.com/brown9804/ML_DS_path/main/_docs/img/azure-machine-learning-taxonomy.png)

![Azure ML environment taxonomy](https://raw.githubusercontent.com/brown9804/ML_DS_path/main/_docs/img/azure-ml-environment-taxonomy.png)

## Typical setup (from scratch)

```console
conda env create --name aml-env --file ./dependencies/environment.yml --force
conda activate aml-env
pip install -r ./dependencies/requirements.txt
```

## Validation checklist

1. Confirm critical libraries are installed.
2. Confirm Python version is what your project expects.
3. Confirm notebook kernel points to the same environment.

Validation:

```console
pip show scikit-learn
pip show azureml-sdk
conda env list
```

Optional kernel registration:

```console
python -m ipykernel install --user --name aml-env --display-name "AML Env"
```

## Common setup failures and fixes

| Symptom | Likely cause | Fix |
|---|---|---|
| Package import error | Dependency missing or version mismatch | Reinstall pinned version from requirements |
| Different results across machines | Unpinned dependencies | Pin versions in environment files |
| Notebook using wrong interpreter | Kernel mismatch | Re-select kernel and restart |
| `conda activate` has no effect | Conda not initialised in shell | Run `conda init bash` (or `zsh`), then reopen terminal |
| pip installs to wrong env | Virtualenv active but pip resolves globally | Use `python -m pip install` instead of bare `pip` |
| Azure ML job uses wrong image | Environment not registered before job submit | Register env first or use `Environment.from_conda_specification` |

## Azure ML environment registration

Registering a local environment to Azure ML so it can be used in remote training jobs:

```python
from azureml.core import Workspace, Environment

ws = Workspace.from_config()
env = Environment.from_conda_specification(
    name="fraud-train",
    file_path="./environment.yml"
)
env.register(workspace=ws)
```

After registration, reference it in job config by name and version:

```python
from azureml.core import ScriptRunConfig
from azureml.core.runconfig import RunConfiguration

rc = RunConfiguration()
rc.environment = Environment.get(ws, name="fraud-train", version="1")

config = ScriptRunConfig(
    source_directory="./src",
    script="train.py",
    run_config=rc,
    compute_target="gpu-cluster"
)
```

## Conda vs pip vs Docker: when to use each

| Tool | Best for | Avoid when |
|---|---|---|
| Conda | Mixed Python + native library deps | Simple pure-Python projects |
| pip + venv | Pure Python projects | Complex C/CUDA dependencies |
| Docker | Full system reproducibility | Team unfamiliar with containers |
| Azure ML curated images | Standard frameworks (PyTorch, TF) | Custom low-level system libs |

These references help when sizing compute and understanding memory/number representation
concepts that affect performance decisions.

![Binary vs decimal data measurements](https://raw.githubusercontent.com/brown9804/ML_DS_path/main/_docs/img/binary_vs_decimal_data_measurements.png)

![Summary of number systems](https://raw.githubusercontent.com/brown9804/ML_DS_path/main/_docs/img/summary_of_number_system.png)

## Quick self-check

1. Why should train and inference share a pinned environment?
2. What command shows all conda environments?
3. When should you register a Jupyter kernel?
