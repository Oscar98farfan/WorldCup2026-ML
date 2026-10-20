# 🏆 WorldCup2026-ML

Modelo de Machine Learning para predecir los resultados del **Mundial de Fútbol 2026** (USA, México y Canadá).

Predice partido a partido y simula el torneo completo para estimar qué selecciones llegan a octavos, cuartos, semifinales y la final.

---

## ¿Qué hace este proyecto?

1. **Limpia y prepara** los datos históricos de partidos mundiales (1930-2022) y el ranking FIFA histórico
2. **Entrena un modelo** Gradient Boosting que predice el resultado de cada partido
3. **Simula el torneo 2026** 10,000 veces partido a partido
4. **Genera predicciones** con probabilidades de avance por ronda para cada selección

---

## Estructura del proyecto

```
WorldCup2026-ML/
│
├── data/
│   ├── raw/                    ← Datos originales (no modificar)
│   │   ├── matches.csv         ← Partidos históricos de mundiales
│   │   ├── rankings.csv        ← Ranking FIFA histórico (1992-2023)
│   │   └── groups_2026.csv     ← Grupos del Mundial 2026
│   └── processed/              ← Generado automáticamente
│
├── models/                     ← Modelos entrenados (.pkl)
├── outputs/                    ← Predicciones y gráficas generadas
│   └── plots/
│
├── src/                        ← Módulos reutilizables
│   ├── data_loader.py          ← Carga de archivos
│   ├── feature_engineering.py  ← Construcción de features
│   └── simulator.py            ← Simulador del torneo
│
├── notebooks/                  ← Análisis exploratorio
│
├── 01_etl.py                   ← PASO 1: Limpieza y preparación
├── 02_train.py                 ← PASO 2: Entrenamiento del modelo
├── 03_simulate.py              ← PASO 3: Simulación del torneo
├── 04_report.py                ← PASO 4: Gráficas y reportes
├── main.py                     ← Ejecuta todo en orden
│
├── requirements.txt
└── README.md
```

---

## Cómo ejecutar

### 1. Clonar el repositorio
```bash
git clone https://github.com/tu-usuario/WorldCup2026-ML.git
cd WorldCup2026-ML
```

### 2. Crear entorno virtual e instalar dependencias
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Ejecutar el pipeline completo
```bash
python main.py
```

### O ejecutar paso a paso
```bash
python 01_etl.py        # Limpieza de datos
python 02_train.py      # Entrenamiento del modelo
python 03_simulate.py   # Simulación del torneo
python 04_report.py     # Gráficas y reportes
```

---

## Datos

| Archivo | Descripción | Fuente |
|---|---|---|
| `matches.csv` | Partidos de mundiales 1930-2022 | Compilación histórica |
| `rankings.csv` | Ranking FIFA mensual 1992-2023 | FIFA |
| `groups_2026.csv` | Grupos confirmados del Mundial 2026 | FIFA (actualización pendiente) |

> **Nota:** 6 equipos están pendientes de confirmación (play-offs). Se usaron placeholders temporales que se actualizarán cuando se confirmen los clasificados.

---

## Modelo

**Algoritmo:** Gradient Boosting  
**Target:** `gana_partido` — si el equipo ganó ese partido (1) o no (0)  
**Validación:** Walk-Forward (predecir cada mundial entrenando solo con mundiales anteriores)

**Features (19):**
- Del partido: `goals`, `penalty`
- Del ranking FIFA: `rank`, `points`, `rank_diff`, `points_diff`, `rank_ratio`, `same_conf`, `is_home`, `is_host`, `rank_tier`
- Históricas del equipo: `hist_mundiales`, `hist_max_ronda`, `hist_avg_ronda`, `hist_veces_qf`, `hist_veces_sf`, `hist_veces_final`, `hist_avg_goles`, `hist_tendencia`

**Métricas (Walk-Forward 2006-2022):**
- AUC promedio: 0.678 ± 0.129
- F1-macro promedio: 0.606 ± 0.090

---

## Resultados — Mundial 2026

*(Se actualizan al ejecutar el pipeline)*

Ver `outputs/simulation_results.csv` y `outputs/plots/`

---

## Autor

Oscar Daniel Farfan Juanias  

---

## Notas técnicas

- Los datos `processed/` y `models/` no se suben a Git (ver `.gitignore`). Se regeneran ejecutando los scripts.
- Para actualizar los equipos de play-offs, editar `data/raw/groups_2026.csv` y volver a ejecutar desde `03_simulate.py`.
- Compatible con Python 3.9+
