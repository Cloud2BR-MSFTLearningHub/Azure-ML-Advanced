
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

> **Note - What this shows:** The workspace taxonomy again, here to emphasize *where environments live*. The environment you
> build locally becomes a registered, versioned asset inside this structure so remote jobs can
> reuse it.

![Azure ML environment taxonomy](https://raw.githubusercontent.com/brown9804/ML_DS_path/main/_docs/img/azure-ml-environment-taxonomy.png)

> **Note - What this shows:** How one environment definition flows into both training and inference jobs. Pinning it once and
> reusing it is the core mechanism behind reproducible runs and deterministic rebuilds.

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
kernel : preventing the common "notebook uses the wrong environment" bug.

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
: the run record then points at an immutable environment version, not a mutable local machine.

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

> **Note - What this shows:** The difference between binary (1 KiB = 1024 bytes) and decimal (1 KB = 1000 bytes) measures.
> It matters when sizing datasets, memory, and compute : a mismatch explains many "why is my data
> bigger than expected?" surprises.

![Summary of number systems](https://raw.githubusercontent.com/brown9804/ML_DS_path/main/_docs/img/summary_of_number_system.png)

> **Note - What this shows:** A summary of number systems (binary, decimal, hexadecimal). Useful background when reading
> memory addresses, byte sizes, and encoded data formats during environment and data debugging.

## The modern SDK v2 equivalent

The registration snippets above use the v1 SDK (`azureml.core`). New projects should prefer the
v2 SDK (`azure-ai-ml`), which models the environment as a declarative object and is the basis of
the CLI v2 and YAML workflows used across this hub. The concepts are identical: pin dependencies,
build a versioned image, reuse it everywhere.

```python
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Environment
from azure.identity import DefaultAzureCredential

ml_client = MLClient.from_config(DefaultAzureCredential())

env = Environment(
    name="fraud-train",
    description="Pinned training/inference environment",
    conda_file="./environment.yml",
    image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest",
)
ml_client.environments.create_or_update(env)
```

The same environment can be declared as YAML and version-controlled alongside your code, which
is the recommended pattern for auditable, GitOps-style pipelines:

```yaml
# environment.yml (Azure ML CLI v2 asset)
$schema: https://azuremlschemas.azureedge.net/latest/environment.schema.json
name: fraud-train
image: mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest
conda_file: ./conda.yml
description: Pinned training/inference environment
```

> **Tip - v1 vs v2:** If you see `from azureml.core import ...` you are on the v1 SDK; if you see
> `from azure.ai.ml import ...` you are on v2. Pick one per project and stay consistent. v2 is the
> forward-looking choice and aligns with the CLI/YAML examples in the deployment modules.

## End-to-end verification script

Run this short script after building an environment to fail fast on the most common problems:
wrong Python version, missing libraries, and non-deterministic seeds. Catching these locally is
far cheaper than discovering them inside a remote job.

```python
import sys, importlib

# 1) Python version must match what the project pins
assert sys.version_info[:2] == (3, 10), f"Expected Python 3.10, got {sys.version}"

# 2) Critical libraries must import at the pinned versions
expected = {"sklearn": "1.3.0", "pandas": "2.0.3", "lightgbm": "4.0.0"}
for mod, want in expected.items():
    got = importlib.import_module(mod).__version__
    assert got == want, f"{mod}: expected {want}, got {got}"

# 3) Seeds must make a run reproducible
import numpy as np
np.random.seed(42)
first = np.random.rand(3)
np.random.seed(42)
assert np.allclose(first, np.random.rand(3)), "Seeding is not deterministic"

print("Environment verification passed.")
```

## Quick self-check

1. Why should train and inference share a pinned environment?
2. What command shows all conda environments?
3. When should you register a Jupyter kernel?
4. How do you tell whether a code sample uses SDK v1 or v2?
5. Which dependency layers (OS, runtime, packages, seeds) must be pinned for full reproducibility?

