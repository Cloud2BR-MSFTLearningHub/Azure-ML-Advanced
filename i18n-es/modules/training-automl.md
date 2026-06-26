# Entrenamiento y AutoML

Este módulo explica cómo se entrenan los modelos en Azure ML, qué hace AutoML en segundo plano,
y cómo pasar de experimentos baseline a una selección de modelos confiable.

## Objetivos de aprendizaje

1. Comprender el entrenamiento manual frente a AutoML.
2. Configurar una ejecución de AutoML con restricciones útiles.
3. Interpretar los resultados de las ejecuciones y elegir un candidato para producción.

![Diagrama de AutoML](https://raw.githubusercontent.com/brown9804/ML_DS_path/main/_docs/img/automl_diagram.png)

> **Nota - Qué muestra esto:** El ciclo de AutoML: probar combinaciones de algoritmo + hiperparámetros, evaluar cada una con validación cruzada,
> y clasificarlas según la métrica principal. AutoML no inventa algoritmos : asigna un presupuesto de búsqueda fijo
> entre algoritmos conocidos.

![Expectativas del proceso de AutoML](https://raw.githubusercontent.com/brown9804/ML_DS_path/main/_docs/img/automl_process_what_to_expect.png)

> **Consejo - Qué esperar:** Las primeras pruebas son baselines débiles; la calidad aumenta a medida que la búsqueda explora, y el ganador final es
> a menudo un ensamble de votación/apilamiento de las mejores ejecuciones. Reserva suficientes iteraciones antes de confiar en la
> tabla de clasificación.

![Pasos detallados de un pronóstico de series temporales basado en ML](https://raw.githubusercontent.com/brown9804/ML_DS_path/main/_docs/img/Detailed-steps-of-ML-based-time-series-forecast.png)

> **Nota - Qué muestra esto:** Los pasos detallados de un pronóstico de series temporales basado en ML. Observa la validación *rolling-origin*:
> un k-fold simple filtraría valores futuros, por lo que los datos temporales se validan avanzando a través del tiempo.

## Flujo de trabajo de AutoML

1. Elegir el tipo de tarea
2. Proporcionar los datos de entrenamiento
3. Seleccionar el destino de cómputo
4. Establecer la métrica y las restricciones
5. Enviar la ejecución y comparar candidatos

## Qué hace AutoML en segundo plano

- Prueba múltiples algoritmos e hiperparámetros.
- Ejecuta la evaluación con validación cruzada/validación.
- Aplica transformaciones de características cuando se configuran.
- Registra métricas, artefactos y linaje.
- Devuelve la mejor ejecución/modelo según la métrica principal elegida.

### Modelos candidatos de AutoML (clasificación tabular)

AutoML normalmente evalúa algunos o todos los siguientes:

| Modelo candidato | Notas |
|---|---|
| LightGBM | A menudo el mejor en datos tabulares; rápido y eficiente en memoria |
| XGBoost | Fuerte competidor; más hiperparámetros |
| LogisticRegression | Baseline rápido; revela si una estructura lineal es suficiente |
| RandomForest | Buena estabilidad, menos ajuste |
| ExtraTrees | Variante de entrenamiento más rápida del random forest |
| Voting Ensemble | Ensamble específico de AutoML de las mejores ejecuciones |
| Stack Ensemble | Meta-modelo específico de AutoML sobre las mejores ejecuciones |

El `VotingEnsemble` o `StackEnsemble` al final es la forma en que AutoML obtiene rendimiento adicional más allá de los modelos individuales, y a menudo son el ganador final.

## AutoML frente a entrenamiento manual: cuándo usar cada uno

AutoML es una opción predeterminada potente, pero no siempre es la herramienta adecuada. La elección se trata de cuánto
control del dominio necesitas frente a cuánta búsqueda quieres automatizar.

| Usa AutoML cuando | Prefiere el entrenamiento manual cuando |
|---|---|
| Necesitas un baseline sólido rápidamente | Tienes una arquitectura específica en mente (por ejemplo, una red neuronal personalizada) |
| El problema es tabular/de pronóstico estándar | Necesitas control total del bucle de entrenamiento o de la función de pérdida |
| Quieres que se maneje la featurización segura frente a fugas | Requieres ingeniería de características a medida o lógica de CV personalizada |
| Quieres comparar muchos algoritmos objetivamente | El presupuesto de cómputo es ajustado y la familia de modelos ya está decidida |

En la práctica, muchos equipos usan ambos: AutoML para descubrir un candidato sólido y validar la
precisión alcanzable, y luego un pipeline construido a mano para refinar, optimizar la latencia y llevarlo a producción.

## Cómputo y rendimiento

Relación de rendimiento:

$$
\text{Performance}=\frac{1}{\text{Execution Time}}
$$
El tiempo de ejecución se ve afectado por:

- El volumen de datos y la dimensionalidad de las características
- La complejidad del algoritmo
- El tamaño del cómputo (CPU/GPU, memoria)
- La paralelización y el máximo de iteraciones concurrentes

## Lista de verificación de configuración mínima de AutoML

| Configuración | Por qué importa |
|---|---|
| task | Define la familia de modelos candidatos |
| primary metric | Alinea la optimización con el objetivo de negocio |
| iterations/timeout | Controla el presupuesto de búsqueda |
| cross-validation | Mejora la robustez de la clasificación |
| featurization settings | Afecta la calidad del modelo y la reproducibilidad |

### Ejemplo mínimo de código de AutoML (Azure SDK v2)

```python
from azure.ai.ml import MLClient, automl
from azure.ai.ml.entities import AmlCompute
from azure.identity import DefaultAzureCredential

ml_client = MLClient(
    credential=DefaultAzureCredential(),
    subscription_id="<sub-id>",
    resource_group_name="<rg>",
    workspace_name="<ws>"
)

classification_job = automl.classification(
    compute="cpu-cluster",
    experiment_name="fraud-automl",
    training_data=ml_client.data.get("fraud-train", version="1"),
    target_column_name="is_fraud",
    primary_metric="AUC_weighted",
    n_cross_validations=5,
    enable_model_explainability=True,
    timeout_minutes=60,
    max_concurrent_trials=4,
)

returned_job = ml_client.jobs.create_or_update(classification_job)
```

Indicadores clave:
- `AUC_weighted` es más seguro que `accuracy` para el fraude (clases desbalanceadas).
- `enable_model_explainability=True` genera la importancia de características basada en SHAP.
- `max_concurrent_trials` debe coincidir con el número de núcleos del clúster de cómputo.

## Errores comunes

- Elegir accuracy para una clasificación desbalanceada.
- Ejecutar muy pocas iteraciones y confiar demasiado en el primer ganador.
- Ignorar la latencia/el costo al seleccionar la mejor puntuación.

## Diseño del espacio de búsqueda (importante)

La calidad de AutoML depende del espacio de búsqueda, no solo del número de iteraciones.

| Parámetro | Demasiado estrecho | Demasiado amplio | Enfoque práctico |
|---|---|---|---|
| Familias de modelos | Omite un mejor tipo de modelo | Desperdicia presupuesto | Empieza amplio, poda después del baseline |
| Tasa de aprendizaje | Puede omitir el punto óptimo de convergencia | Exploración lenta | Usa rangos en escala logarítmica |
| Profundidad/hojas del árbol | Riesgo de underfit | Riesgo de overfit + latencia | Restringe según el presupuesto de latencia |
| Regularización | Ajuste de ruido sub-regularizado | Underfit sobre-regularizado | Ajusta con CV y verificaciones de holdout |

## Opciones de estrategia de validación

| Contexto | Enfoque de validación |
|---|---|
| Tabular estándar | Validación cruzada K-fold |
| Pronóstico temporal | Validación rolling-origin |
| Entidades agrupadas | Divisiones de entidades tipo GroupKFold |

## Campos de seguimiento de experimentos que se deben persistir

Metadatos mínimos para la reproducibilidad:

- ID de ejecución, ID de ejecución principal
- Instantánea/versión del código
- Versión del activo del conjunto de datos
- Versión del entorno
- Conjunto de características/hash
- Hiperparámetros
- Métricas por división
- URI/versión del modelo de salida

## Política de selección de candidatos

Selecciona el candidato de despliegue usando criterios multiobjetivo:

$$
\text{Score}_{deploy}=w_1\cdot\text{Quality}-w_2\cdot\text{Latency}-w_3\cdot\text{Cost}+w_4\cdot\text{Stability}
$$

donde los pesos $w_i$ reflejan las prioridades del negocio.

## Puertas de promoción (de dev a prod)

1. Se cumple el umbral de la métrica offline.
2. La latencia de inferencia está por debajo del SLO en hardware representativo.
3. Se aprueban el análisis de seguridad y la política de dependencias.
4. Se completa la revisión de explicabilidad/equidad.
5. Se registra la aprobación del flujo de trabajo de autorización.

## Autoevaluación rápida

1. ¿Por qué es crítica la elección de la métrica principal en AutoML?
2. ¿Qué compensación controla el máximo de iteraciones concurrentes?
3. ¿Por qué deben considerarse las restricciones de despliegue durante la selección del modelo?

## Análisis a fondo: cada concepto, explicado

Esta sección explica qué automatiza AutoML, qué *no* hace, y por qué existe cada control.

### Qué busca realmente AutoML

AutoML es una búsqueda estructurada sobre tres elecciones acopladas: **featurización** (cómo las columnas en bruto
se convierten en entradas del modelo), **algoritmo** (qué familia de modelos) e **hiperparámetros** (las configuraciones
dentro de esa familia). Conceptualmente está resolviendo una optimización externa:

$$
\min_{a \in \text{algorithms},\; h \in \text{hyperparams}(a)}\; \text{ValidationLoss}(a, h)
$$

No inventa nuevos algoritmos : *asigna inteligentemente un presupuesto fijo* de pruebas entre
algoritmos conocidos, usando los resultados obtenidos hasta el momento para decidir qué probar a continuación. Por eso el "diseño del espacio de búsqueda"
importa más que el número bruto de iteraciones: un buen espacio contiene la región ganadora; uno malo nunca
lo hace.

### La featurización, desmitificada

Cuando está habilitada, AutoML maneja automáticamente la imputación de valores faltantes, la codificación de categorías,
la vectorización de texto y el escalado de características : los mismos pasos del módulo de preparación de datos, aplicados
de forma consistente dentro de los pliegues de la validación cruzada para que no filtren información. El beneficio es un preprocesamiento seguro frente a fugas y
reproducible; el costo es menos control manual, razón por la cual las configuraciones de `featurization`
son explícitas y se registran para la reproducibilidad.

### La validación cruzada dentro de AutoML y por qué clasifica los modelos de forma justa

`n_cross_validations=5` significa que cada candidato se evalúa en 5 pliegues de validación rotativos y los
resultados se promedian. Esto reduce la posibilidad de que una división afortunada corone al modelo equivocado. Para
datos **temporales**, un k-fold simple filtra el futuro, por lo que en su lugar se usa la validación **rolling-origin**;
para entidades **agrupadas** (por ejemplo, múltiples filas por cliente), las divisiones conscientes del grupo evitan
que la misma entidad aparezca tanto en entrenamiento como en validación.

### Métrica principal: alinear el optimizador con el negocio

AutoML optimiza exactamente una **métrica principal**, por lo que elegirla *es* elegir qué significa "el mejor".
En problemas desbalanceados, `accuracy` es engañosa (un modelo que predice "nunca fraude" obtiene 99%),
por lo que en su lugar se usan `AUC_weighted` o `average_precision`. La lección se generaliza: el optimizador
explotará sin piedad cualquier métrica que le des, por lo que la métrica debe codificar la estructura de costos
real.

### Ensambles: por qué el ganador suele ser un `VotingEnsemble`

Después de probar modelos individuales, AutoML construye dos meta-modelos:

- **Ensamble de votación** : promedia las predicciones de las mejores ejecuciones. Los modelos diversos cometen
  errores *no correlacionados*, por lo que el promedio es más preciso y estable que cualquier miembro individual.
- **Ensamble de apilamiento** : entrena un pequeño meta-modelo sobre las predicciones fuera de pliegue de los modelos base para
  aprender *cómo* combinarlos.

Estos suelen ganar porque combinar aprendices diversos reduce la varianza : el mismo principio de bagging/stacking
del módulo de tipos de modelos, aplicado automáticamente.

### Concurrencia, presupuesto y la compensación entre costo y tiempo

`max_concurrent_trials` controla cuántos candidatos se entrenan en paralelo; establecerlo en el
número de nodos del clúster mantiene el cómputo ocupado y acorta el tiempo de reloj, pero **no** reduce
el costo total de cómputo (pagas por el mismo número de pruebas, solo que más rápido). `timeout_minutes` y
los límites de iteración acotan el **presupuesto** de búsqueda : la perilla central que compensa la exhaustividad frente al
tiempo y el dinero.

### La puntuación de selección multiobjetivo, explicada

La puntuación del candidato $\text{Score}_{deploy}=w_1\text{Quality}-w_2\text{Latency}-w_3\text{Cost}+w_4\text{Stability}$
formaliza una verdad del mundo real: el modelo desplegable maximiza la calidad *y* la estabilidad mientras se
penaliza por la latencia y el costo. Los pesos $w_i$ codifican las prioridades del negocio : una API en tiempo real
pondera fuertemente la latencia; un trabajo por lotes nocturno la pondera cerca de cero. AutoML clasifica según la métrica
principal, pero la decisión de promoción *humana* debería usar este objetivo más completo, que es exactamente por lo que
las **puertas de promoción** verifican la latencia-bajo-SLO, la seguridad y la equidad, no solo la puntuación offline.

### Por qué los metadatos de seguimiento de experimentos no son negociables

La lista de campos que se deben persistir (ID de ejecución, versión de datos, versión del entorno, hash de características,
hiperparámetros, métricas por división, URI del modelo) es lo que hace que un resultado sea **reproducible** y
**auditable**. Si no puedes responder "¿qué datos, código y entorno produjeron este modelo?", no
puedes depurar una regresión, pasar una auditoría ni reentrenar de forma segura : por eso estos metadatos son la columna vertebral del
MLOps, no un registro opcional.
