
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

> Image explanation: This visual shows azure ml workspace taxonomy. Use it to understand the concept in this section and connect it to practical Azure ML decisions.

![Azure ML environment taxonomy](https://raw.githubusercontent.com/brown9804/ML_DS_path/main/_docs/img/azure-ml-environment-taxonomy.png)

> Image explanation: This visual shows azure ml environment taxonomy. Use it to understand the concept in this section and connect it to practical Azure ML decisions.

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

## Deep dive: every concept, explained

This section explains the moving parts of a reproducible environment and why each one can
silently change model behavior.

### What "environment" actually contains

An ML environment is a stack of layers, and a mismatch in *any* layer can change results:

| Layer | Example | Failure if it drifts |
|---|---|---|
| OS / system libraries | glibc, CUDA, BLAS | Numeric differences, GPU ops fail |
| Language runtime | Python 3.10 vs 3.11 | Syntax/ABI breaks, pickled models won't load |
| Packages | scikit-learn 1.3 vs 1.4 | Different defaults → different predictions |
| Random seeds | NumPy/PyTorch seed | Non-deterministic training runs |

Reproducibility means pinning *all* of these, which is why Azure ML packages them into a single
**versioned environment** (a container image) rather than relying on whatever is installed on a
machine.

### conda vs pip vs the environment files

- **conda** manages *both* Python and non-Python system dependencies (CUDA, MKL, compilers),
  which is why it is preferred for the base environment in data science.
- **pip** installs Python packages from PyPI; it does not manage system libraries.
- `environment.yml` declares the conda environment (channels + packages); `requirements.txt`
  pins pip packages installed *into* that environment. Using both lets conda handle the heavy
  system layer and pip handle pure-Python packages.

### Why `python -m pip` instead of bare `pip`

`pip` is just a script that points at *some* Python. If multiple Pythons exist, bare `pip` can
install into the wrong one. `python -m pip` runs pip *as a module of the exact interpreter you
invoked*, guaranteeing the package lands in the environment you think it does. The same logic
applies to `python -m ipykernel install`, which registers *this* interpreter as a notebook
kernel — preventing the common "notebook uses the wrong environment" bug.

### Pinning, lockfiles, and determinism

- **Pinning** means specifying exact versions (`scikit-learn==1.3.2`) instead of ranges
  (`scikit-learn>=1.3`). Ranges let a rebuild silently pull a newer package whose changed
  defaults alter predictions.
- A **lockfile** captures the *entire resolved dependency tree* (including transitive
  dependencies) so a rebuild is byte-for-byte reproducible. This is what auditors and incident
  responders rely on to recreate a past model exactly.

### Registering an environment to Azure ML

`Environment.from_conda_specification(...).register(workspace=ws)` builds a container image from
your spec and stores it as a *versioned* asset in the workspace. The benefit: the **same image**
is reused across remote training jobs and the inference deployment, eliminating training/serving
skew. Referencing it by `name` + `version` in `ScriptRunConfig` makes the run fully reproducible
— the run record then points at an immutable environment version, not a mutable local machine.

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

> Image explanation: This visual shows binary vs decimal data measurements. Use it to understand the concept in this section and connect it to practical Azure ML decisions.

![Summary of number systems](https://raw.githubusercontent.com/brown9804/ML_DS_path/main/_docs/img/summary_of_number_system.png)

> Image explanation: This visual shows summary of number systems. Use it to understand the concept in this section and connect it to practical Azure ML decisions.

## Quick self-check

1. Why should train and inference share a pinned environment?
2. What command shows all conda environments?
3. When should you register a Jupyter kernel?

