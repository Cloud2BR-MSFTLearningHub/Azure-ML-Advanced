# Métricas de rendimiento

Elegir la métrica correcta es una de las decisiones más importantes en ML. Un modelo puede verse
excelente en una métrica y fallar en el objetivo de negocio real.

![Cómo elegir una métrica](https://raw.githubusercontent.com/brown9804/ML_DS_path/main/_docs/img/How-to-Choose-a-Metric-for-Imbalanced-Classification-latest.png)

> **Consejo - Cómo usar este gráfico:** Elige la métrica según el *costo de los errores*, no por costumbre. En problemas desbalanceados prefiere F1, PR-AUC,
> o MCC sobre accuracy; pondera el recall cuando los positivos no detectados son costosos, la precisión cuando las falsas
> alarmas son costosas.

## Conceptos básicos de la matriz de confusión

Conteos:

- $TP$: verdaderos positivos
- $FP$: falsos positivos
- $TN$: verdaderos negativos
- $FN$: falsos negativos

Tasas:

$$
\text{TPR}=\frac{TP}{TP+FN},\quad
\text{FPR}=\frac{FP}{FP+TN},\quad
\text{TNR}=\frac{TN}{TN+FP}
$$

### Lectura de la matriz de confusión

```
Actual \\ Predicted |  Positive  |  Negative
---------------------|------------|----------
       Positive      |   TP       |   FN
       Negative      |   FP       |   TN
```

| Celda | Significado | Ejemplo (modelo de fraude) |
|---|---|---|
| TP | Positivo marcado correctamente | Fraude real detectado |
| FP | Falsa alarma | Transacción legítima bloqueada |
| TN | Negativo descartado correctamente | Transacción legítima aprobada |
| FN | Positivo no detectado | Fraude que se escapa |

Para un caso de uso de fraude, **FN es el error más peligroso** (fraude no detectado). Esto significa que el recall debe ponderarse fuertemente en la elección de la métrica.

## Métricas de clasificación

- Precisión: $\frac{TP}{TP+FP}$
- Recall: $\frac{TP}{TP+FN}$
- F1: $2\cdot\frac{PR}{P+R}$
- AUC

Fórmulas adicionales:

$$
\text{Accuracy}=\frac{TP+TN}{TP+TN+FP+FN}
$$

$$
\mathrm{MCC}=\frac{TP\cdot TN-FP\cdot FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}
$$

$$
\mathrm{AUC}=\int_0^1 \mathrm{TPR}(\mathrm{FPR})\,d\mathrm{FPR}
$$

Cuándo usar cada una:

| Escenario | Mejores opciones de métrica |
|---|---|
| Desbalance de clases | F1, PR-AUC, MCC, balanced accuracy |
| Alto costo de falsos negativos | Recall, F2 |
| Alto costo de falsos positivos | Precisión |
| Calidad de probabilidad | Log loss, métricas de calibración |

## Métricas de regresión

- MAE
- RMSE
- R2

Fórmulas:

$$
\mathrm{MAE}=\frac{1}{N}\sum_{i=1}^{N}|y_i-\hat{y}_i|,
\quad
\mathrm{RMSE}=\sqrt{\frac{1}{N}\sum_{i=1}^{N}(y_i-\hat{y}_i)^2}
$$

$$
R^2=1-\frac{\sum_{i=1}^{N}(y_i-\hat{y}_i)^2}{\sum_{i=1}^{N}(y_i-\bar{y})^2}
$$

Consejos de interpretación:

- MAE es robusta y fácil de explicar en las unidades originales.
- RMSE penaliza los errores grandes con más fuerza.
- $R^2$ compara contra un baseline de predicción de la media.

## Métricas de pronóstico (práctico)

- MAPE: error porcentual intuitivo, inestable cerca de valores cero.
- sMAPE: variante simétrica para una mejor comparabilidad.
- RMSE/MAE: siguen siendo útiles para la calidad del pronóstico.

Fórmulas:

$$
\text{MAPE}=\frac{100}{N}\sum_{i=1}^{N}\left|\frac{y_i-\hat{y}_i}{y_i}\right|
$$

$$
\text{sMAPE}=\frac{100}{N}\sum_{i=1}^{N}\frac{2|y_i-\hat{y}_i|}{|y_i|+|\hat{y}_i|}
$$

Orientación: prefiere RMSE/MAE para comparar modelos en la misma escala. Usa MAPE/sMAPE solo cuando comuniques errores como porcentajes a las partes interesadas del negocio.

## Errores a evitar

- Reportar una métrica sin intervalos de confianza.
- Comparar modelos en diferentes divisiones de validación.
- Ignorar el ajuste del umbral en la clasificación.
- Usar accuracy como la métrica principal para datos desbalanceados.
- Evaluar solo globalmente cuando el rendimiento a nivel de segmento puede divergir significativamente.

## Optimización del umbral (clasificación)

Las decisiones de clasificación en producción requieren una política de umbral, no el $0.5$ predeterminado.

Costo de negocio esperado en el umbral $\tau$:

$$
\mathbb{E}[\text{Cost}(\tau)] = C_{FP}\cdot FP(\tau)+C_{FN}\cdot FN(\tau)
$$

Elige el $\tau$ que minimice el costo esperado bajo las restricciones del negocio.

## Calibración y confiabilidad

Un clasificador puede clasificar bien (alto AUC) pero producir probabilidades mal calibradas.

Verificaciones de calibración:

- Curva de confiabilidad
- Brier score
- Error de calibración esperado (ECE)

Usa métodos de calibración (Platt scaling, regresión isotónica) cuando los sistemas de decisión dependan
de los valores de probabilidad, no solo del orden de clasificación.

## Ejemplos de SLI/SLO para la calidad del modelo

| SLI | Ejemplo de SLO |
|---|---|
| macro-F1 semanal | >= 0.82 |
| RMSE semanal | <= 5.0 |
| Error de calibración | <= 0.03 |
| Ratio de disparidad por segmento | <= 1.25 |

## Autoevaluación rápida

1. ¿Qué métrica es más segura que accuracy para datos desbalanceados?
2. ¿Por qué RMSE puede ser mucho mayor que MAE?
3. ¿Qué implica un $R^2$ negativo?

## Análisis a fondo: cada concepto, explicado

Esta sección explica *por qué* cada métrica está construida como está y cuándo engaña.

### La matriz de confusión es la fuente de cada métrica de clasificación

Todas las métricas de clasificación son proporciones de las cuatro celdas $TP, FP, TN, FN$. Memorizar las celdas
es suficiente para reconstruir cualquier métrica:

- **Precisión** $\tfrac{TP}{TP+FP}$ responde "cuando el modelo dice positivo, ¿con qué frecuencia acierta?"
  : es la métrica que te importa cuando los **falsos positivos son costosos** (bloquear buenos
  clientes, marcar pacientes sanos).
- **Recall / TPR** $\tfrac{TP}{TP+FN}$ responde "de todos los positivos reales, ¿cuántos detectamos?" :
  la métrica cuando los **falsos negativos son costosos** (fraude no detectado, enfermedad no detectada).
- **F1** $2\tfrac{PR}{P+R}$ es la **media armónica** de las dos. La media armónica (no el
  promedio ordinario) se usa porque se mantiene baja a menos que *ambas*, precisión y recall, sean altas : se
  niega a recompensar a un modelo que sacrifica una por la otra.

### Por qué accuracy falla en datos desbalanceados

Accuracy $\tfrac{TP+TN}{\text{all}}$ pondera cada ejemplo por igual, así que cuando el 99% de los casos son
negativos, predecir "siempre negativo" obtiene 99% mientras detecta cero positivos. Por eso el
curso dirige repetidamente hacia **F1, PR-AUC, MCC o balanced accuracy** para problemas sesgados :
todas, de diferentes maneras, evitan que la clase mayoritaria domine la puntuación.

### ROC-AUC frente a PR-AUC, y qué significa "libre de umbral"

- **AUC** es el área bajo la curva ROC (TPR frente a FPR a medida que el umbral barre de 1 a 0).
  Equivale a la probabilidad de que el modelo clasifique un positivo aleatorio por encima de un negativo aleatorio : una medida pura
  de **calidad de clasificación**, independiente de cualquier umbral elegido.
- En un desbalance fuerte, ROC-AUC puede verse engañosamente alto porque el enorme conteo de negativos mantiene baja la FPR.
  **PR-AUC** (precisión frente a recall) se centra en la clase positiva y es el resumen más honesto
  cuando los positivos son raros.
- **MCC** (coeficiente de correlación de Matthews) usa las cuatro celdas en un único número balanceado de
  −1 a +1, razón por la cual es robusto incluso bajo un desbalance severo.

### La optimización del umbral como minimización de costos

Un modelo produce probabilidades; el **umbral** $\tau$ las convierte en decisiones. Como los falsos
positivos y los falsos negativos suelen tener costos *diferentes*, el umbral óptimo minimiza el
costo esperado $\mathbb{E}[\text{Cost}(\tau)] = C_{FP}\cdot FP(\tau) + C_{FN}\cdot FN(\tau)$ en lugar
de maximizar accuracy. Concretamente: si un fraude no detectado cuesta 20 veces más que una falsa alarma, reduces $\tau$
para cambiar muchos falsos positivos por menos falsos negativos. El 0.5 predeterminado casi nunca es óptimo
en producción.

### Métricas de regresión: MAE frente a RMSE frente a $R^2$

- **MAE** promedia los errores absolutos : está en las unidades del objetivo y trata todos los errores linealmente,
  por lo que es **robusta a valores atípicos** y fácil de explicar ("desviado en \$5 en promedio").
- **RMSE** promedia los errores *al cuadrado* y luego saca la raíz cuadrada, así que los errores grandes se penalizan
  desproporcionadamente. RMSE ≥ MAE siempre; una *gran brecha* entre ellas señala unos pocos grandes fallos
  (errores de cola pesada) que vale la pena investigar.
- $R^2 = 1 - \tfrac{SS_{res}}{SS_{tot}}$ compara el modelo contra el baseline trivial de "predecir la media".
  $R^2=1$ es perfecto, $0$ significa no mejor que la media, y **un $R^2$ negativo significa que el
  modelo es *peor* que predecir la media** : una señal clara de que algo está roto.

### Métricas de pronóstico y la trampa del denominador cero

**MAPE** expresa el error como porcentaje, lo que las partes interesadas encuentran intuitivo, pero **divide por
el valor real**, por lo que se dispara o queda indefinido cerca de cero y penaliza en exceso las subestimaciones.
**sMAPE** simetriza el denominador para acotar el valor y tratar las sobre/subestimaciones de forma más
equitativa. La regla práctica del módulo: optimiza y compara modelos con RMSE/MAE (estables),
y traduce a MAPE/sMAPE solo para *comunicación*.

### Calibración: clasificar bien no es lo mismo que acertar

Un modelo con alto AUC clasifica los ejemplos correctamente pero aún puede producir probabilidades que no
coinciden con la realidad (por ejemplo, sus predicciones de "90%" son correctas solo el 70% de las veces). Cuando las decisiones
posteriores usan el *valor* de la probabilidad (cálculos de pérdida esperada, fijación de precios, triaje), necesitas
**calibración**:

- La **curva de confiabilidad** grafica la frecuencia predicha frente a la observada; la diagonal es perfecta.
- El **Brier score** es el error cuadrático medio de las probabilidades en sí.
- El **ECE** (error de calibración esperado) resume la brecha promedio entre la confianza y la accuracy.
- **Platt scaling** (ajustar una logística sobre las puntuaciones) y la **regresión isotónica** (ajustar una función escalonada
  monótona) son las correcciones post-hoc estándar.

### De las métricas a los SLI/SLO : cerrando el ciclo hacia las operaciones

Una métrica offline se convierte en un **SLI** de producción (indicador de nivel de servicio) cuando se mide
de forma continua, y en un **SLO** (objetivo) cuando se le adjunta un umbral (por ejemplo, "macro-F1 semanal ≥
0.82"). Así es como la calidad del modelo se une a la latencia y la disponibilidad como una propiedad monitoreada y
alertable : el puente desde este módulo hacia el monitoreo de drift y los SLO de despliegue más adelante en el
curso.

## Autoevaluación rápida (análisis a fondo)

1. ¿Por qué F1 es la media armónica de la precisión y el recall en lugar del promedio ordinario?
2. En un conjunto de datos 99% negativo, ¿por qué ROC-AUC puede verse genial mientras PR-AUC es pobre?
3. ¿Qué te dice un $R^2$ negativo sobre el modelo?
4. ¿Por qué el umbral predeterminado de 0.5 casi nunca es óptimo en producción?
5. Un modelo tiene un AUC alto pero sus predicciones de "90%" son correctas solo el 70% de las veces: ¿cuál es el problema y qué correcciones aplican?
