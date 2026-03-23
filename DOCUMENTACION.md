# WorldCup2026-ML — Documentación Completa

**Autor:** Oscar Daniel Farfan Juanias  
**Proyecto:** Predicción del Mundial de Fútbol 2026 usando Machine Learning

---

## ¿Qué hace este proyecto en palabras simples?

Imagina que quieres apostar quién va a ganar el Mundial 2026. No puedes adivinar, pero sí puedes estudiar la historia. Eso es exactamente lo que hace este proyecto:

1. **Estudia** todos los partidos de mundiales desde 1930 hasta 2022
2. **Aprende** qué equipos suelen ganar sus partidos y por qué
3. **Simula** el Mundial 2026 miles de veces usando lo que aprendió
4. **Entrega** una tabla con probabilidades reales para cada equipo

El resultado no dice "Francia va a ganar". Dice "Francia tiene 3.3% de probabilidad de ser campeón". La diferencia es importante: el fútbol tiene azar, y el modelo lo respeta.

---

## ¿Por qué se necesita Machine Learning para esto?

El fútbol parece simple: el mejor equipo gana. Pero la realidad es más compleja. En el Mundial 2018 Alemania (campeón defensor) quedó eliminada en la fase de grupos. En 2022 Marruecos llegó a semifinales siendo un equipo "pequeño".

Un humano que intente predecir manualmente tendría que analizar 48 equipos, sus historiales, sus rankings actuales, quiénes son sus rivales en el grupo, etc. El Machine Learning hace eso de forma sistemática, sin sesgos, con todos los datos históricos disponibles.

---

## ¿Qué datos se usaron?

### Fuente 1: `matches.csv` (partidos históricos)
Contiene **964 partidos** de mundiales desde 1930 hasta 2022. Por cada partido sabe:
- Qué equipos jugaron
- Cuántos goles hizo cada uno
- Si hubo penales
- En qué ronda fue el partido
- Quién era el anfitrión del mundial

### Fuente 2: `rankings.csv` (ranking FIFA histórico)
Contiene **67,463 registros** del ranking FIFA desde diciembre 1992 hasta junio 2024. Por cada fecha y país sabe:
- En qué posición estaba ese país
- Cuántos puntos FIFA tenía
- Si subió o bajó respecto al período anterior

**¿Por qué desde 1992?** Porque el ranking FIFA se creó en agosto de 1993. Para mundiales anteriores a 1994 el modelo usa un valor por defecto (posición 75).

### Fuente 3: `groups_2026.csv` (grupos del Mundial 2026)
Los 48 equipos organizados en 12 grupos. 42 confirmados oficialmente y 6 placeholders pendientes de play-offs.

---

## El corazón del proyecto: las 19 variables (features)

El modelo no trabaja con texto ni con videos. Trabaja con números. Para cada partido se calculan 19 variables que describen la situación del equipo:

### Grupo 1 — Del partido en sí

| Variable | Qué mide | Ejemplo |
|---|---|---|
| `goals` | Goles anotados en ese partido | Brasil anota 3 goles → goals=3 |
| `penalty` | Penales anotados | Si ganaron en penales → penalty=1 |

### Grupo 2 — Del ranking FIFA

| Variable | Qué mide | Ejemplo |
|---|---|---|
| `rank` | Posición en el ranking FIFA antes del mundial | Argentina rank=1 |
| `points` | Puntos FIFA acumulados | Argentina points=1855 |
| `rank_diff` | Diferencia de posición con el rival | Si Argentina (1) vs Haití (55): rank_diff=-54 |
| `points_diff` | Diferencia de puntos con el rival | 1855 - 800 = +1055 |
| `rank_ratio` | Qué tan mejor/peor es el rival en términos relativos | Rival rank / Tu rank |
| `same_conf` | ¿Son de la misma confederación? | Brasil vs Argentina = 1 (ambos CONMEBOL) |
| `is_home` | ¿Juega como local en ese partido? | 1 si es el primer equipo listado |
| `is_host` | ¿Es el anfitrión del mundial? | USA/México/Canadá = 1 en 2026 |
| `rank_tier` | Categoría del equipo | Top 10=3, Top 30=2, Top 60=1, resto=0 |

### Grupo 3 — Historial en mundiales anteriores

| Variable | Qué mide | Ejemplo |
|---|---|---|
| `hist_mundiales` | Cuántos mundiales jugó antes | Brasil ha jugado 22 mundiales |
| `hist_max_ronda` | Mejor ronda histórica | Brasil llegó a la final (4) |
| `hist_avg_ronda` | Ronda promedio en toda su historia | Brasil promedia 2.27 por mundial |
| `hist_veces_qf` | Veces que llegó a cuartos o más | Brasil 16 veces |
| `hist_veces_sf` | Veces que llegó a semis o más | Brasil 12 veces |
| `hist_veces_final` | Veces que llegó a la final | Brasil 7 veces |
| `hist_avg_goles` | Promedio de goles histórico | Brasil ~2 goles por partido |
| `hist_tendencia` | Promedio de los últimos 2 mundiales | ¿Está en ascenso o descenso? |

**¿Por qué el historial es tan importante?** Porque captura algo que el ranking no puede: la *tradición mundialista*. Suiza tiene un ranking alto pero históricamente no llega lejos en mundiales. Brasil tiene un historial excepcional. Esa diferencia importa.

---

## ¿Qué NO se incluyó y por qué?

### Attendance (asistencia al estadio) — EXCLUIDA
Durante el desarrollo se descubrió que la asistencia tenía correlación con la ronda del partido, no con la calidad del equipo. Las finales siempre tienen más público simplemente porque son finales. Al incluirla, el modelo aprendía algo falso: "más público = equipo mejor". Además, para predecir 2026 no sabemos de antemano en qué estadio jugará cada partido.

### xG (expected goals) — EXCLUIDO
Los expected goals miden la calidad de las ocasiones de gol. Al calcular la importancia de variables, xG tuvo importancia 0.000, es decir, no aportaba ninguna información adicional al modelo. Incluirla solo añadiría ruido.

---

## El modelo: Gradient Boosting

### ¿Qué es Gradient Boosting?

Imagina que quieres aprender a predecir partidos. Contratas a un primer experto, que hace su mejor predicción pero comete errores. Luego contratas un segundo experto que se especializa en corregir los errores del primero. Luego un tercero que corrige los errores de los dos anteriores. Y así 200 veces.

Eso es Gradient Boosting: 200 "árboles de decisión" pequeños construidos en secuencia, donde cada uno aprende de los errores del anterior.

### ¿Por qué Gradient Boosting y no otro algoritmo?

Durante el desarrollo se compararon tres algoritmos:

| Algoritmo | AUC | F1-macro |
|---|---|---|
| Regresión Logística | 0.688 | 0.727 |
| Random Forest | 0.728 | 0.750 |
| **Gradient Boosting** | **0.835** | **0.833** |

Gradient Boosting ganó por amplio margen. Es especialmente bueno con datasets pequeños y variables mixtas (números de distintas escalas), exactamente como este proyecto.

### ¿Qué predice exactamente el modelo?

El modelo predice: **¿Este equipo va a ganar este partido específico?**

Respuesta: una probabilidad entre 0% y 100%.

Por ejemplo: "Brasil tiene 72% de probabilidad de ganar a Marruecos".

Este es el motor que impulsa la simulación. Para cada partido, el modelo calcula cuánta probabilidad tiene cada equipo de ganar, y con eso se "lanza el dado".

---

## La validación: Walk-Forward

### ¿Por qué validar con Walk-Forward?

El problema con evaluar un modelo de datos temporales (como mundiales) es que no puedes mezclar el futuro con el pasado. Si usas datos de 2022 para predecir 2010, estás haciendo trampa.

Walk-Forward respeta el tiempo:
- Para predecir 2006 → el modelo solo ve datos hasta 2005
- Para predecir 2010 → el modelo solo ve datos hasta 2009
- Para predecir 2014 → el modelo solo ve datos hasta 2013
- Para predecir 2018 → el modelo solo ve datos hasta 2017
- Para predecir 2022 → el modelo solo ve datos hasta 2021

Esto simula exactamente cómo funcionaría en la realidad.

### Resultados de la validación

```
Año    Train    Test    AUC    F1-macro   Accuracy
2006   1,288    128    0.934    0.835      0.844
2010   1,416    128    0.864    0.747      0.758
2014   1,544    128    0.937    0.859      0.859
2018   1,672    128    0.846    0.760      0.766
2022   1,800    128    0.896    0.801      0.805
─────────────────────────────────────────────────
PROM                   0.895    0.800      0.806
```

**¿Qué significa AUC=0.895?**
AUC mide qué tan bien el modelo distingue entre equipos que ganan y equipos que pierden. Un valor de 0.5 sería adivinar al azar (lanzar una moneda). Un valor de 1.0 sería perfecto. Con 0.895 el modelo clasifica correctamente el 89.5% de los pares equipo-ganador vs equipo-perdedor.

**¿Qué significa F1-macro=0.800?**
F1 equilibra la precisión (cuándo dice que un equipo gana, ¿acierta?) con el recall (¿detecta todos los equipos que realmente ganan?). Un valor de 0.800 es muy bueno para este tipo de problema.

**¿Por qué 2018 es el año más bajo (AUC=0.846)?**
El Mundial 2018 fue históricamente sorpresivo: Alemania eliminada en grupos, Argentina y Brasil cayendo temprano, Bélgica llegando a semis. Ningún modelo basado en historial puede predecir esas sorpresas. Eso es una limitación inherente del fútbol, no del modelo.

---

## Dos modelos: Evaluación y Producción

Este es uno de los aspectos más importantes del diseño. Se entrenan dos modelos distintos:

### Modelo de Evaluación
- **Entrena con:** datos < 2022 (1930 hasta 2021)
- **Evalúa en:** datos = 2022 (Qatar)
- **Propósito:** responder "¿funciona este enfoque?"
- **Se guarda:** NO (solo sirve para medir)

### Modelo de Producción
- **Entrena con:** datos <= 2022 (1930 hasta 2022 inclusive)
- **Propósito:** predecir el Mundial 2026
- **Se guarda:** SÍ → `models/match_predictor.pkl`

**¿Por qué incluir 2022 en el modelo de producción?**

Imagina que eres médico y aprendes medicina con libros hasta 2021. Luego tratas pacientes de 2026. ¿No sería mejor también haber estudiado los casos de 2022 y 2023? Claro que sí.

Lo mismo aquí: Qatar 2022 tiene información valiosa. Marruecos llegó a semis. Argentina ganó el título. Croacia llegó a semis de nuevo. Alemania quedó en grupos. Todo eso es información que le dice al modelo cómo es el fútbol moderno.

El modelo de evaluación ya confirmó que el enfoque funciona bien. Entonces para predecir 2026 le damos al modelo toda la información disponible, incluyendo 2022.

---

## La simulación: Monte Carlo

### ¿Qué es Monte Carlo?

Monte Carlo es una técnica matemática que resuelve problemas difíciles usando aleatoriedad. El nombre viene del famoso casino de Mónaco, porque usa el azar de forma controlada.

**Ejemplo simple:** Quieres saber cuánto mide el radio de un círculo sin usar la fórmula π. Tiras 10,000 dardos al azar dentro de un cuadrado. Cuentas cuántos cayeron dentro del círculo inscrito. La proporción te da una estimación de π. Cuantos más dardos, más preciso.

### Cómo funciona en este proyecto

En lugar de dardos, usamos simulaciones del torneo completo **respetando el fixture oficial**:

**Simulación número 1:**
1. **Fase de grupos:** cada equipo juega solo contra los 3 de su grupo.
   - Grupo C: Brasil vs Marruecos → el modelo dice Brasil tiene 72% → Brasil gana
   - Grupo C: Brasil vs Haití → el modelo dice Brasil tiene 85% → Brasil gana
   - Grupo C: Brasil vs Escocia → Brasil gana → Brasil pasa 1ro del Grupo C
   - Marruecos también pasa como 2do del Grupo C
2. **Octavos:** los cruces siguen el fixture oficial de FIFA.
   - Brasil (1ro Grupo C) enfrenta al 2do del Grupo D — **nunca a Marruecos**
   - Marruecos (2do Grupo C) enfrenta al 1ro del Grupo D — **nunca a Brasil**
   - Brasil y Marruecos solo podrían encontrarse en la **final**
3. **Cuartos, semis y final:** el bracket mantiene separados a los equipos
   que venían de partes distintas del cuadro
4. En esta simulación, digamos que Argentina gana el mundial

**Simulación número 2:**
1. Los mismos partidos, pero esta vez los dados caen diferente
2. México pierde contra Corea del Sur
3. ... (diferentes resultados)
4. En esta simulación, Francia gana el mundial

Después de 1,000 simulaciones, contamos:
- ¿En cuántas ganó Argentina? → 49 → 4.9% de probabilidad
- ¿En cuántas ganó Bélgica? → 85 → 8.5% de probabilidad

### ¿Por qué 1,000 simulaciones y no más?

Con 10,000 simulaciones los porcentajes son más precisos (diferencias de décimas). Con 1,000 ya son lo suficientemente confiables para identificar los favoritos. La diferencia entre 8.5% y 8.3% no cambia la interpretación: Bélgica es favorita.

Se usaron 1,000 porque la PC tardó ~52 minutos en terminar. Con 10,000 hubiera tardado ~8-9 horas.

---

## Cómo leer los resultados

### La tabla principal

```
Pos  Equipo        Octavos  Cuartos  Semis   Final  Campeón
  1  Belgium        91.0%   58.5%   38.5%   25.8%    8.5%
  2  Germany        94.5%   63.5%   38.3%   23.4%    7.6%
  3  Switzerland    94.9%   58.5%   34.1%   21.5%    6.6%
...
 43  Brazil         87.0%   35.0%   14.6%    5.0%    0.4%
 47  Italy          26.0%    7.0%    1.5%    0.2%    0.2%
```

**Cómo leerla de izquierda a derecha:**

Cada columna es una ronda más avanzada. Los números siempre van de mayor a menor de izquierda a derecha porque pasar una ronda es requisito para llegar a la siguiente.

Ejemplo con Bélgica:
- **91.0% Octavos** → En 910 de 1,000 simulaciones Bélgica pasó la fase de grupos
- **58.5% Cuartos** → En 585 simulaciones llegó a cuartos de final
- **38.5% Semis** → En 385 simulaciones llegó a semifinales
- **25.8% Final** → En 258 simulaciones llegó a la final
- **8.5% Campeón** → En 85 simulaciones Bélgica ganó el mundial

**¿Cómo comparar dos equipos?**

Brasil tiene 87% de pasar grupos pero solo 0.4% de ganar. Eso dice que Brasil casi siempre pasa de grupos pero frecuentemente cae antes de las fases finales. Bélgica tiene 91% de pasar grupos y 8.5% de ganar, lo que indica que cuando Bélgica avanza, suele ir muy lejos.

### Las gráficas generadas

**`predictions_by_round.png` — Gráfica principal**

Una barra horizontal por equipo, ordenados de mayor a menor probabilidad de llegar a semis. Los colores representan rondas:
- Verde pálido = llega a octavos pero no más
- Verde claro = llega a cuartos
- Verde oscuro = llega a semis
- Azul = llega a la final
- Azul oscuro = campeón

Un equipo con mucho azul oscuro es el gran favorito. Un equipo donde solo hay verde pálido es uno que suele pasar grupos pero cae pronto.

**`top_candidates.png` — Top 16 candidatos**

Muestra los 16 equipos con mayor probabilidad de avanzar. Para cada equipo hay 5 barras juntas, una por ronda. Permite comparar directamente cómo se "va filtrando" cada equipo a medida que avanza el torneo.

Un equipo cuyas 5 barras bajan muy rápido (la de octavos es alta pero la de final es muy baja) es un equipo que suele llegar lejos pero raramente gana el torneo completo.

**`group_analysis.png` — Análisis por grupo**

12 gráficas, una por grupo. Muestra la probabilidad de pasar grupos para cada equipo del grupo. Los equipos en naranja son placeholders (pendientes de confirmar).

Un grupo donde un equipo tiene 90%+ y los demás tienen 30-40% es un "grupo fácil" con un claro favorito. Un grupo donde todos tienen 50-60% es un "grupo de la muerte" muy parejo.

---

## Resultados y qué nos dicen

### Los favoritos del modelo

**Top 3 candidatos al título:**
1. **Bélgica 8.5%** — Grupo favorable (Egipto, Irán, Nueva Zelanda) + buen ranking (3 mundial)
2. **Alemania 7.6%** — Historial excepcional + grupo manejable
3. **Suiza 6.6%** — Grupo muy asequible + ranking sólido

### Las anomalías que vale la pena entender

**Brasil posición 43 con solo 0.4% de ser campeón (resultado con simulador anterior):**
Este resultado era incorrecto porque el simulador anterior mezclaba aleatoriamente los clasificados en octavos. Brasil (Grupo C) y Marruecos (también Grupo C) se encontraban en muchas simulaciones en octavos, cuando en la realidad nunca pueden enfrentarse antes de la final. Eso perjudicaba artificialmente a Brasil. Con el simulador corregido (fixture oficial), los resultados de Brasil mejorarán significativamente al no enfrentar a Marruecos hasta fases avanzadas.

**Bélgica como favorita:**
Bélgica tiene ranking 3 del mundo y quedó en el **Grupo G con Egipto, Irán y Nueva Zelanda**, un grupo relativamente asequible. El modelo detecta esto y le da buenas probabilidades de llegar lejos.

**Italia posición 47 con 0.2%:**
Italia es placeholder en el Grupo B. Si efectivamente Italia clasificó por el play-off, tiene el historial y el ranking para hacerlo mucho mejor. Este resultado es una consecuencia directa de ser placeholder — su confirmación cambiaría los números.

### Lo que el modelo NO puede predecir

Honestamente, el modelo tiene limitaciones claras:

1. **Forma actual del equipo:** ¿Tienen lesionados clave? ¿Vienen de una buena racha? El modelo no sabe nada de eso.

2. **Sorpresas como 2018:** Alemania eliminada en grupos. Argentina cayendo temprano. Esas situaciones son casi imposibles de predecir con historial y ranking.

3. **Equipos nuevos o con pocos datos:** Canadá tiene 2 mundiales en el dataset (0 victorias). El modelo no sabe bien qué esperar de ellos.

4. **Cambios recientes:** Un equipo que cambió de entrenador en 2024 sigue siendo evaluado con el historial de la era anterior.

---

## Por qué se separó el proyecto en archivos distintos

El proyecto tiene 7 archivos de código. Esto no es capricho: es una decisión de diseño con razones concretas.

### `src/data_loader.py` — El portero de los datos
Centraliza toda la lectura de archivos. Si mañana `matches.csv` se renombra a `world_cup_matches.csv`, solo cambias una línea aquí y todo el resto sigue funcionando. Sin este módulo, habría rutas de archivo repetidas en 5 scripts distintos.

### `src/feature_engineering.py` — El cerebro de las transformaciones
Todo el código que convierte datos crudos en variables útiles para el modelo está aquí. Tanto `01_etl.py` como `03_simulate.py` lo usan. Si hubiera que cambiar cómo se calcula el historial de un equipo, se cambia en un solo lugar.

### `src/simulator.py` — El motor de la simulación
La lógica del torneo (grupos, octavos, cuartos, semis, final) está separada de los scripts principales. Respeta el fixture oficial del Mundial 2026: los cruces de octavos están predefinidos según grupo y posición, garantizando que equipos del mismo grupo nunca se enfrenten antes de la final. Si en el futuro quisieras cambiar las reglas, solo modificas este archivo.

### `01_etl.py` — Limpieza y preparación
Lee los datos crudos y produce el dataset limpio. Si consigues datos nuevos, solo corres este script y todo lo demás se actualiza.

### `02_train.py` — Entrenamiento
Entrena dos modelos: uno para evaluar (medir qué tan bueno es) y otro para producción (predecir el 2026). Guarda el modelo en disco para que `03_simulate.py` no tenga que reentrenar cada vez.

### `03_simulate.py` — Simulación
Carga el modelo guardado y simula el torneo. Al estar separado del entrenamiento, puedes correrlo mil veces con diferentes configuraciones sin reentrenar.

### `04_report.py` — Reportes y gráficas
Lee los resultados y genera las visualizaciones. Puedes cambiar colores, agregar gráficas nuevas o modificar el formato sin tocar la simulación.

### `main.py` — El director de orquesta
Ejecuta todo en orden. Para un usuario nuevo, `python main.py` es todo lo que necesita saber.

---

## Cómo actualizar el proyecto para el 2026

### Cuando se confirmen los 6 equipos de play-offs

Edita `data/raw/groups_2026.csv` y reemplaza los placeholders:

```
Antes: Denmark, no, Play-off D (Czech Republic/Denmark)
Después: Czech Republic, yes,
```

Luego corre:
```bash
python 03_simulate.py
python 04_report.py
```

No necesitas reentrenar el modelo. Solo re-simular.

### Cuando quieras actualizar el ranking FIFA

Descarga el ranking más reciente de FIFA y agrégalo a `rankings.csv`. Luego corre el pipeline completo:
```bash
python main.py
```

---

## Resumen del pipeline completo

```
matches.csv ─────────────────────────────────────┐
rankings.csv ────── 01_etl.py ─── model_dataset.csv ─── 02_train.py ─── match_predictor.pkl
groups_2026.csv ─────────────────────────────────┘              │
                                                                  ▼
                                              03_simulate.py ─── simulation_results.csv
                                                                  │
                                              04_report.py ───── predictions_by_round.png
                                                                  top_candidates.png
                                                                  group_analysis.png
                                                                  report_2026.txt
```

---

## Glosario de términos técnicos

| Término | Significado en palabras simples |
|---|---|
| **Machine Learning** | Un programa que aprende de ejemplos en lugar de seguir reglas escritas a mano |
| **Gradient Boosting** | Algoritmo que construye 200 modelos pequeños en secuencia, cada uno corrigiendo los errores del anterior |
| **Feature** | Una variable de entrada del modelo. Son las 19 columnas que describen cada partido |
| **Target** | Lo que el modelo predice: si el equipo ganó el partido (1) o no (0) |
| **AUC** | Métrica de 0 a 1. Qué tan bien distingue el modelo entre ganadores y perdedores. 0.5=azar, 1.0=perfecto |
| **F1-macro** | Métrica que equilibra precisión y recall. 0.8 es muy bueno para este tipo de problema |
| **Walk-Forward** | Técnica de validación que respeta el orden temporal. No usa el futuro para predecir el pasado |
| **Data Leakage** | Error grave donde el modelo "ve" información futura durante el entrenamiento |
| **Monte Carlo** | Técnica que resuelve problemas complejos usando simulaciones aleatorias repetidas |
| **Overfitting** | Cuando el modelo memoriza los datos de entrenamiento pero falla en datos nuevos |
| **Pickle (.pkl)** | Formato para guardar un modelo Python en disco y cargarlo después sin reentrenar |
| **Scaler** | Herramienta que normaliza los números para que el modelo los compare en la misma escala |
| **Placeholder** | Equipo temporal que ocupa el lugar de uno que aún no está confirmado |
