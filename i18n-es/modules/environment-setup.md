# Configuración del entorno

Este módulo ayuda a las personas principiantes a construir un entorno de ejecución reproducible
desde cero. La reproducibilidad es crítica porque el comportamiento del modelo depende de las
versiones de los paquetes, las bibliotecas del sistema operativo y los detalles del entorno de
ejecución de Python.

## Por qué importa la reproducibilidad del entorno

- El mismo código puede producir resultados diferentes con versiones distintas de dependencias.
- El entrenamiento y la inferencia deben compartir bibliotecas compatibles.
- Los equipos necesitan reconstrucciones deterministas para auditorías y recuperación ante incidentes.

Los diagramas siguientes muestran cómo se organizan los activos de Azure ML y cómo se reutilizan
los entornos en el entrenamiento y la inferencia.

![Taxonomía del área de trabajo de Azure ML](https://raw.githubusercontent.com/brown9804/ML_DS_path/main/_docs/img/azure-machine-learning-taxonomy.png)

> **Nota - Lo que muestra:** Nuevamente la taxonomía del área de trabajo, aquí para enfatizar *dónde viven los entornos*. El entorno que
> construyes localmente se convierte en un activo registrado y versionado dentro de esta estructura para que los trabajos remotos puedan
> reutilizarlo.

![Taxonomía del entorno de Azure ML](https://raw.githubusercontent.com/brown9804/ML_DS_path/main/_docs/img/azure-ml-environment-taxonomy.png)

> **Nota - Lo que muestra:** Cómo una definición de entorno fluye tanto hacia los trabajos de entrenamiento como de inferencia. Fijarla una vez y
> reutilizarla es el mecanismo central detrás de las ejecuciones reproducibles y las reconstrucciones deterministas.

## Configuración típica (desde cero)

```console
conda env create --name aml-env --file ./dependencies/environment.yml --force
conda activate aml-env
pip install -r ./dependencies/requirements.txt
```

## Lista de verificación de validación

1. Confirma que las bibliotecas críticas están instaladas.
2. Confirma que la versión de Python es la que espera tu proyecto.
3. Confirma que el kernel del notebook apunta al mismo entorno.

Validación:

```console
pip show scikit-learn
pip show azureml-sdk
conda env list
```

Registro opcional del kernel:

```console
python -m ipykernel install --user --name aml-env --display-name "AML Env"
```

## Fallos comunes de configuración y soluciones

| Síntoma | Causa probable | Solución |
|---|---|---|
| Error de importación de paquete | Dependencia faltante o discrepancia de versión | Reinstalar la versión fijada desde requirements |
| Resultados diferentes entre máquinas | Dependencias sin fijar | Fijar versiones en los archivos de entorno |
| Notebook usa el intérprete equivocado | Discrepancia de kernel | Volver a seleccionar el kernel y reiniciar |
| `conda activate` no tiene efecto | Conda no inicializado en el shell | Ejecutar `conda init bash` (o `zsh`), luego reabrir la terminal |
| pip instala en el entorno equivocado | Virtualenv activo pero pip resuelve globalmente | Usar `python -m pip install` en lugar de `pip` simple |
| El trabajo de Azure ML usa la imagen equivocada | Entorno no registrado antes de enviar el trabajo | Registrar el entorno primero o usar `Environment.from_conda_specification` |

## Registro del entorno en Azure ML

Registrar un entorno local en Azure ML para que pueda usarse en trabajos de entrenamiento remotos:

```python
from azureml.core import Workspace, Environment

ws = Workspace.from_config()
env = Environment.from_conda_specification(
    name="fraud-train",
    file_path="./environment.yml"
)
env.register(workspace=ws)
```

Después del registro, refiérete a él en la configuración del trabajo por nombre y versión:

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

## Análisis a fondo: cada concepto, explicado

Esta sección explica las piezas móviles de un entorno reproducible y por qué cada una puede
cambiar silenciosamente el comportamiento del modelo.

### Qué contiene realmente un "entorno"

Un entorno de ML es una pila de capas, y una discrepancia en *cualquier* capa puede cambiar los resultados:

| Capa | Ejemplo | Fallo si se desvía |
|---|---|---|
| Bibliotecas del sistema operativo | glibc, CUDA, BLAS | Diferencias numéricas, fallos en operaciones de GPU |
| Entorno de ejecución del lenguaje | Python 3.10 vs 3.11 | Rupturas de sintaxis/ABI, los modelos serializados no cargan |
| Paquetes | scikit-learn 1.3 vs 1.4 | Valores predeterminados distintos → predicciones distintas |
| Semillas aleatorias | Semilla de NumPy/PyTorch | Ejecuciones de entrenamiento no deterministas |

La reproducibilidad significa fijar *todos* estos elementos, por lo que Azure ML los empaqueta en un único
**entorno versionado** (una imagen de contenedor) en lugar de depender de lo que esté instalado en una
máquina.

### conda vs pip vs los archivos de entorno

- **conda** gestiona *tanto* Python como las dependencias de sistema no relacionadas con Python (CUDA, MKL, compiladores),
  por lo que es preferible para el entorno base en ciencia de datos.
- **pip** instala paquetes de Python desde PyPI; no gestiona bibliotecas del sistema.
- `environment.yml` declara el entorno de conda (canales + paquetes); `requirements.txt`
  fija los paquetes de pip instalados *dentro* de ese entorno. Usar ambos permite que conda gestione la pesada
  capa del sistema y que pip gestione los paquetes puros de Python.

### Por qué `python -m pip` en lugar de `pip` simple

`pip` es solo un script que apunta a *algún* Python. Si existen varios Python, el `pip` simple puede
instalar en el equivocado. `python -m pip` ejecuta pip *como un módulo del intérprete exacto que
invocaste*, garantizando que el paquete quede en el entorno que crees. La misma lógica
se aplica a `python -m ipykernel install`, que registra *este* intérprete como un kernel de
notebook : evitando el error común de "el notebook usa el entorno equivocado".

### Fijado, archivos de bloqueo y determinismo

- **Fijar** significa especificar versiones exactas (`scikit-learn==1.3.2`) en lugar de rangos
  (`scikit-learn>=1.3`). Los rangos permiten que una reconstrucción extraiga silenciosamente un paquete más nuevo cuyos
  valores predeterminados cambiados alteran las predicciones.
- Un **archivo de bloqueo** captura el *árbol de dependencias resuelto completo* (incluidas las
  dependencias transitivas) para que una reconstrucción sea reproducible bit a bit. Esto es en lo que confían los auditores
  y los responsables de incidentes para recrear un modelo pasado de forma exacta.

### Registrar un entorno en Azure ML

`Environment.from_conda_specification(...).register(workspace=ws)` construye una imagen de contenedor a partir de
tu especificación y la almacena como un activo *versionado* en el área de trabajo. La ventaja: la **misma imagen**
se reutiliza en los trabajos de entrenamiento remotos y en el despliegue de inferencia, eliminando el sesgo entre
entrenamiento y servicio. Referenciarla por `name` + `version` en `ScriptRunConfig` hace que la ejecución sea totalmente reproducible
: el registro de la ejecución apunta entonces a una versión de entorno inmutable, no a una máquina local mutable.

## Conda vs pip vs Docker: cuándo usar cada uno

| Herramienta | Mejor para | Evitar cuando |
|---|---|---|
| Conda | Dependencias mixtas de Python + bibliotecas nativas | Proyectos simples puros de Python |
| pip + venv | Proyectos puros de Python | Dependencias complejas de C/CUDA |
| Docker | Reproducibilidad completa del sistema | El equipo no está familiarizado con contenedores |
| Imágenes curadas de Azure ML | Frameworks estándar (PyTorch, TF) | Bibliotecas de sistema personalizadas de bajo nivel |

Estas referencias ayudan al dimensionar el cómputo y al comprender los conceptos de memoria/representación
numérica que afectan las decisiones de rendimiento.

![Mediciones de datos binarias vs decimales](https://raw.githubusercontent.com/brown9804/ML_DS_path/main/_docs/img/binary_vs_decimal_data_measurements.png)

> **Nota - Lo que muestra:** La diferencia entre medidas binarias (1 KiB = 1024 bytes) y decimales (1 KB = 1000 bytes).
> Importa al dimensionar conjuntos de datos, memoria y cómputo : una discrepancia explica muchas sorpresas del tipo "¿por qué mis datos
> son más grandes de lo esperado?".

![Resumen de los sistemas numéricos](https://raw.githubusercontent.com/brown9804/ML_DS_path/main/_docs/img/summary_of_number_system.png)

> **Nota - Lo que muestra:** Un resumen de los sistemas numéricos (binario, decimal, hexadecimal). Conocimiento de fondo útil al leer
> direcciones de memoria, tamaños de bytes y formatos de datos codificados durante la depuración del entorno y los datos.

## El equivalente moderno del SDK v2

Los fragmentos de registro anteriores usan el SDK v1 (`azureml.core`). Los proyectos nuevos deberían preferir el
SDK v2 (`azure-ai-ml`), que modela el entorno como un objeto declarativo y es la base de
la CLI v2 y los flujos de trabajo YAML usados en todo este centro. Los conceptos son idénticos: fijar dependencias,
construir una imagen versionada, reutilizarla en todas partes.

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

El mismo entorno puede declararse como YAML y controlarse por versiones junto a tu código, lo cual
es el patrón recomendado para pipelines auditables al estilo GitOps:

```yaml
# environment.yml (Azure ML CLI v2 asset)
$schema: https://azuremlschemas.azureedge.net/latest/environment.schema.json
name: fraud-train
image: mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest
conda_file: ./conda.yml
description: Pinned training/inference environment
```

> **Consejo - v1 vs v2:** Si ves `from azureml.core import ...` estás en el SDK v1; si ves
> `from azure.ai.ml import ...` estás en v2. Elige uno por proyecto y mantén la consistencia. v2 es la
> opción orientada al futuro y se alinea con los ejemplos de CLI/YAML en los módulos de despliegue.

## Script de verificación de extremo a extremo

Ejecuta este breve script después de construir un entorno para fallar rápido ante los problemas más comunes:
versión de Python equivocada, bibliotecas faltantes y semillas no deterministas. Detectarlos localmente es
mucho más barato que descubrirlos dentro de un trabajo remoto.

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

## Autoevaluación rápida

1. ¿Por qué deberían el entrenamiento y la inferencia compartir un entorno fijado?
2. ¿Qué comando muestra todos los entornos de conda?
3. ¿Cuándo deberías registrar un kernel de Jupyter?
4. ¿Cómo distingues si un ejemplo de código usa el SDK v1 o v2?
5. ¿Qué capas de dependencias (sistema operativo, entorno de ejecución, paquetes, semillas) deben fijarse para una reproducibilidad total?
