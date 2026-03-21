"""
===========================================================
02_train.py — Entrenamiento del Modelo
===========================================================

PROPÓSITO:
    Lee el dataset procesado por 01_etl.py, entrena el modelo
    Gradient Boosting, lo valida con Walk-Forward y lo guarda
    en models/ listo para usar en el simulador.

¿POR QUÉ GRADIENT BOOSTING?
    Durante el desarrollo del proyecto anterior probamos tres
    algoritmos (Regresión Logística, Random Forest, Gradient
    Boosting). Gradient Boosting ganó consistentemente:
    - AUC test 2022:  0.835
    - F1-macro:       0.833
    Construye árboles secuencialmente corrigiendo los errores
    del anterior, lo que lo hace especialmente bueno en
    datasets pequeños con features mixtas como este.

¿QUÉ ES WALK-FORWARD VALIDATION?
    Una validación temporal que respeta el orden cronológico.
    Para predecir 2006 → entrena solo con datos < 2006.
    Para predecir 2010 → entrena solo con datos < 2010.
    Así simulamos exactamente cómo usaríamos el modelo en
    la realidad. El CV estándar mezcla años y es optimista.

TARGET NUEVO vs PROYECTO ANTERIOR:
    Antes:  llega_lejos (¿llegó a cuartos el equipo?)
    Ahora:  gana_partido (¿ganó ese partido específico?)
    Esto permite usar el modelo partido a partido en el
    simulador para predecir cualquier enfrentamiento.

ENTRADA:
    data/processed/model_dataset.csv → generado por 01_etl.py

SALIDA:
    models/match_predictor.pkl → modelo entrenado
    models/scaler.pkl          → escalador StandardScaler
    models/features.pkl        → lista de features usadas
    models/metrics.json        → métricas de validación

CÓMO EJECUTAR:
    python 02_train.py

AUTOR: Oscar Daniel Farfan Juanias
===========================================================
"""

import pandas as pd
import numpy as np
import pickle
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score, f1_score,
    accuracy_score, classification_report,
    confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns

from src.data_loader import load_processed, get_paths
from src.feature_engineering import MODEL_FEATURES, TARGET


# ===========================================================
# CONFIGURACIÓN DEL MODELO
# ===========================================================
# Parámetros del Gradient Boosting definidos aquí para que
# sean fáciles de encontrar y modificar si se necesita.
#
# n_estimators=200: número de árboles. Más árboles = más preciso
#   pero más lento. 200 es un buen balance para este dataset.
# max_depth=3: profundidad máxima de cada árbol. Valores bajos
#   previenen overfitting. 3 es el estándar para GB.
# learning_rate=0.05: qué tan rápido aprende. Valores bajos +
#   muchos árboles = mejor generalización.
# subsample=0.8: usa el 80% de los datos en cada árbol.
#   Añade aleatoriedad y reduce overfitting.
# random_state=42: semilla para reproducibilidad. Con la misma
#   semilla siempre obtienes el mismo modelo.

GB_PARAMS = dict(
    n_estimators=200,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.8,
    random_state=42
)

# Años que se usan como "test" en el walk-forward
# Solo incluimos mundiales con cobertura de ranking FIFA ≥ 80%
WALK_FORWARD_YEARS = [2006, 2010, 2014, 2018, 2022]

# Año que se usa como test final (el más reciente disponible)
TEST_YEAR = 2022


# ===========================================================
# PASO 1: CARGAR DATASET PROCESADO
# ===========================================================

def load_dataset() -> pd.DataFrame:
    """
    Carga el dataset generado por 01_etl.py.

    Si no existe, muestra un mensaje claro indicando que
    hay que correr 01_etl.py primero.

    RETORNA:
        DataFrame con todas las features y el target
    """
    print("\n" + "="*55)
    print("  PASO 1: Cargando dataset procesado")
    print("="*55)

    df = load_processed("model_dataset.csv")

    print(f"\n  Filas:   {len(df)}")
    print(f"  Años:    {sorted(df['year'].unique())}")
    print(f"\n  Distribución del target '{TARGET}':")
    counts = df[TARGET].value_counts()
    for val, count in counts.items():
        label = "Ganó    " if val == 1 else "No ganó"
        print(f"    {label} ({val}): {count:>4}  ({count/len(df)*100:.1f}%)")

    return df


# ===========================================================
# PASO 2: WALK-FORWARD VALIDATION
# ===========================================================

def walk_forward_validation(df: pd.DataFrame) -> dict:
    """
    Evalúa el modelo con validación temporal (Walk-Forward).

    PROCESO:
        Para cada año en WALK_FORWARD_YEARS:
        1. Train: todos los datos de años ANTERIORES a ese año
        2. Test:  solo los datos de ese año específico
        3. Entrenar modelo en train, evaluar en test
        4. Guardar métricas

    ¿POR QUÉ ES IMPORTANTE?
        Un CV estándar mezclaría datos de 2022 con datos de 2006
        para validar 2010, lo cual no tiene sentido: no puedes
        usar el futuro para predecir el pasado.
        Walk-Forward respeta el tiempo y da métricas honestas.

    PARÁMETROS:
        df → DataFrame completo con features y target

    RETORNA:
        dict con métricas por año y promedios
    """
    print("\n" + "="*55)
    print("  PASO 2: Walk-Forward Validation")
    print("="*55)

    results = []

    print(f"\n  {'Año':>6} {'Train':>7} {'Test':>6} "
          f"{'AUC':>7} {'F1-macro':>10} {'Accuracy':>10}")
    print("  " + "-"*50)

    for year_test in WALK_FORWARD_YEARS:
        train_mask = df["year"] < year_test
        test_mask  = df["year"] == year_test

        # Verificar que hay datos suficientes
        if train_mask.sum() < 50 or test_mask.sum() == 0:
            print(f"  {year_test:>6} → datos insuficientes, omitiendo")
            continue

        X_tr = df.loc[train_mask, MODEL_FEATURES].fillna(0)
        X_te = df.loc[test_mask,  MODEL_FEATURES].fillna(0)
        y_tr = df.loc[train_mask, TARGET]
        y_te = df.loc[test_mask,  TARGET]

        # Escalar — IMPORTANTE: fit solo en train, transform en ambos
        # Si hiciéramos fit en todo el dataset, el test "contaminaría"
        # la escala y los resultados serían optimistas
        scaler_wf  = StandardScaler()
        X_tr_sc    = scaler_wf.fit_transform(X_tr)
        X_te_sc    = scaler_wf.transform(X_te)

        # Entrenar y evaluar
        modelo_wf = GradientBoostingClassifier(**GB_PARAMS)
        modelo_wf.fit(X_tr_sc, y_tr)

        y_prob = modelo_wf.predict_proba(X_te_sc)[:, 1]
        y_pred = modelo_wf.predict(X_te_sc)

        # Calcular métricas
        try:
            auc_val = roc_auc_score(y_te, y_prob)
        except ValueError:
            auc_val = float("nan")

        f1_val  = f1_score(y_te, y_pred, average="macro", zero_division=0)
        acc_val = accuracy_score(y_te, y_pred)

        print(f"  {year_test:>6} {train_mask.sum():>7} {test_mask.sum():>6} "
              f"{auc_val:>7.3f} {f1_val:>10.3f} {acc_val:>10.3f}")

        results.append({
            "year":     year_test,
            "n_train":  int(train_mask.sum()),
            "n_test":   int(test_mask.sum()),
            "auc":      auc_val,
            "f1_macro": f1_val,
            "accuracy": acc_val,
        })

    wf_df = pd.DataFrame(results)
    print("  " + "-"*50)
    print(f"  {'PROM':>6} {'':>7} {'':>6} "
          f"{wf_df['auc'].mean():>7.3f} "
          f"{wf_df['f1_macro'].mean():>10.3f} "
          f"{wf_df['accuracy'].mean():>10.3f}")
    print(f"  {'STD':>6} {'':>7} {'':>6} "
          f"{wf_df['auc'].std():>7.3f} "
          f"{wf_df['f1_macro'].std():>10.3f} "
          f"{wf_df['accuracy'].std():>10.3f}")

    return {
        "by_year":      results,
        "mean_auc":     float(wf_df["auc"].mean()),
        "std_auc":      float(wf_df["auc"].std()),
        "mean_f1":      float(wf_df["f1_macro"].mean()),
        "std_f1":       float(wf_df["f1_macro"].std()),
        "mean_acc":     float(wf_df["accuracy"].mean()),
    }


# ===========================================================
# PASO 3: ENTRENAR MODELO FINAL
# ===========================================================

def train_final_model(df: pd.DataFrame):
    """
    Entrena el modelo final usando todos los datos hasta TEST_YEAR.
    Evalúa en TEST_YEAR como test de referencia.

    ¿POR QUÉ ENTRENAR CON DATOS HASTA 2022 Y NO CON TODO?
        Queremos tener un test set "visto" (2022) para reportar
        las métricas finales del modelo. En producción real
        (predicción 2026) el modelo se re-entrena con todos los
        datos disponibles incluyendo 2022.

    PARÁMETROS:
        df → DataFrame completo

    RETORNA:
        modelo    → GradientBoostingClassifier entrenado
        scaler    → StandardScaler ajustado
        métricas  → dict con resultados en test 2022
    """
    print("\n" + "="*55)
    print(f"  PASO 3: Entrenando modelo final (test = {TEST_YEAR})")
    print("="*55)

    train_mask = df["year"] < TEST_YEAR
    test_mask  = df["year"] == TEST_YEAR

    X_train = df.loc[train_mask, MODEL_FEATURES].fillna(0)
    X_test  = df.loc[test_mask,  MODEL_FEATURES].fillna(0)
    y_train = df.loc[train_mask, TARGET]
    y_test  = df.loc[test_mask,  TARGET]

    print(f"\n  Train: {len(X_train)} filas  "
          f"(mundiales {int(df.loc[train_mask,'year'].min())}-{TEST_YEAR-1})")
    print(f"  Test:  {len(X_test)} filas  (mundial {TEST_YEAR})")

    # Escalar
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    # Entrenar
    modelo = GradientBoostingClassifier(**GB_PARAMS)
    modelo.fit(X_train_sc, y_train)

    # Evaluar
    y_prob = modelo.predict_proba(X_test_sc)[:, 1]
    y_pred = modelo.predict(X_test_sc)

    auc_val = roc_auc_score(y_test, y_prob)
    f1_val  = f1_score(y_test, y_pred, average="macro", zero_division=0)
    acc_val = accuracy_score(y_test, y_pred)

    print(f"\n  Métricas en test {TEST_YEAR}:")
    print(f"    AUC:      {auc_val:.3f}")
    print(f"    F1-macro: {f1_val:.3f}")
    print(f"    Accuracy: {acc_val:.3f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['No ganó','Ganó'], zero_division=0)}")

    metrics = {
        "test_year": TEST_YEAR,
        "auc":       float(auc_val),
        "f1_macro":  float(f1_val),
        "accuracy":  float(acc_val),
    }

    return modelo, scaler, metrics, y_test, y_prob, y_pred


# ===========================================================
# PASO 4: IMPORTANCIA DE FEATURES
# ===========================================================

def show_feature_importance(modelo, paths: dict) -> None:
    """
    Muestra y guarda la importancia de cada feature en el modelo.

    ¿QUÉ ES LA IMPORTANCIA DE FEATURES?
        En Gradient Boosting, cada feature recibe un score
        que indica cuánto contribuyó a las predicciones.
        Features con importancia alta son las más decisivas.
        Features con importancia ~0 no aportan nada.

    PARÁMETROS:
        modelo → modelo entrenado
        paths  → rutas del proyecto (para guardar la gráfica)
    """
    print("\n" + "="*55)
    print("  PASO 4: Importancia de features")
    print("="*55)

    imp_df = pd.DataFrame({
        "Feature":     MODEL_FEATURES,
        "Importancia": modelo.feature_importances_
    }).sort_values("Importancia", ascending=False)

    print(f"\n  {'Feature':<25} {'Importancia':>12} {'Barra'}")
    print("  " + "-"*55)
    for _, row in imp_df.iterrows():
        barra = "█" * int(row["Importancia"] * 100)
        print(f"  {row['Feature']:<25} {row['Importancia']:>12.4f}  {barra}")

    # Guardar gráfica
    plots_dir = paths["outputs"] / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 7))
    imp_sorted = imp_df.sort_values("Importancia", ascending=True)
    plt.barh(imp_sorted["Feature"], imp_sorted["Importancia"], color="#3498db")
    plt.xlabel("Importancia")
    plt.title("Importancia de Features — Gradient Boosting")
    plt.tight_layout()
    plt.savefig(plots_dir / "feature_importance.png", bbox_inches="tight", dpi=120)
    plt.close()
    print(f"\n  ✅ Gráfica guardada: outputs/plots/feature_importance.png")


# ===========================================================
# PASO 5: GUARDAR MODELO Y MÉTRICAS
# ===========================================================

def save_model(modelo, scaler, wf_metrics: dict, final_metrics: dict, paths: dict) -> None:
    """
    Guarda el modelo, el scaler, la lista de features y las métricas.

    ¿QUÉ SE GUARDA Y POR QUÉ?

    match_predictor.pkl:
        El modelo entrenado serializado. 'Serializado' significa
        convertido a bytes para guardarlo en disco. Se carga con
        pickle.load() en 03_simulate.py sin necesidad de reentrenar.

    scaler.pkl:
        El escalador StandardScaler ajustado. DEBE guardarse junto
        al modelo porque transforma los datos con la misma escala
        que se usó en el entrenamiento. Sin él las predicciones
        serían incorrectas.

    features.pkl:
        La lista exacta de features en el orden correcto.
        El modelo es sensible al orden de las columnas. Si cambias
        el orden o agregas una columna sin actualizar este archivo,
        las predicciones serán incorrectas.

    metrics.json:
        Las métricas de validación en formato legible. Útil para
        comparar versiones del modelo y documentar el rendimiento.
    """
    print("\n" + "="*55)
    print("  PASO 5: Guardando modelo")
    print("="*55)

    models_dir = paths["models"]
    models_dir.mkdir(parents=True, exist_ok=True)

    # Guardar modelo
    with open(models_dir / "match_predictor.pkl", "wb") as f:
        pickle.dump(modelo, f)
    print(f"\n  ✅ models/match_predictor.pkl")

    # Guardar scaler
    with open(models_dir / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    print(f"  ✅ models/scaler.pkl")

    # Guardar lista de features
    with open(models_dir / "features.pkl", "wb") as f:
        pickle.dump(MODEL_FEATURES, f)
    print(f"  ✅ models/features.pkl")

    # Guardar métricas en JSON (legible por humanos)
    all_metrics = {
        "model":          "GradientBoostingClassifier",
        "parameters":     GB_PARAMS,
        "features":       MODEL_FEATURES,
        "target":         TARGET,
        "walk_forward":   wf_metrics,
        "test_final":     final_metrics,
    }
    with open(models_dir / "metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"  ✅ models/metrics.json")

    print(f"""
  Para cargar el modelo en otro script:

    import pickle
    with open('models/match_predictor.pkl', 'rb') as f:
        modelo = pickle.load(f)
    with open('models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('models/features.pkl', 'rb') as f:
        features = pickle.load(f)
    """)


# ===========================================================
# PASO 6: GRÁFICAS DE EVALUACIÓN
# ===========================================================

def save_evaluation_plots(
    y_test, y_prob, y_pred,
    wf_metrics: dict,
    paths: dict
) -> None:
    """
    Genera y guarda las gráficas de evaluación del modelo.

    GRÁFICAS:
        1. Walk-Forward por año: AUC y F1-macro a través del tiempo
        2. Matriz de confusión en test 2022
        3. Curva ROC en test 2022
    """
    plots_dir = paths["outputs"] / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Walk-Forward por año
    wf_years = [r["year"]     for r in wf_metrics["by_year"]]
    wf_aucs  = [r["auc"]      for r in wf_metrics["by_year"]]
    wf_f1s   = [r["f1_macro"] for r in wf_metrics["by_year"]]

    axes[0].plot(wf_years, wf_aucs, "o-", label="AUC",
                 color="#3498db", linewidth=2, markersize=8)
    axes[0].plot(wf_years, wf_f1s,  "s-", label="F1-macro",
                 color="#e74c3c", linewidth=2, markersize=8)
    axes[0].axhline(0.5, color="gray", linestyle="--",
                    alpha=0.5, label="Azar (0.5)")
    axes[0].axhline(wf_metrics["mean_auc"], color="#3498db",
                    linestyle=":", alpha=0.7,
                    label=f"AUC prom={wf_metrics['mean_auc']:.3f}")
    axes[0].set_title("Walk-Forward Validation por Mundial")
    axes[0].set_xlabel("Año del mundial de prueba")
    axes[0].set_ylabel("Score")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(0.3, 1.0)

    # 2. Matriz de confusión
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[1],
                xticklabels=["No ganó", "Ganó"],
                yticklabels=["No ganó", "Ganó"])
    axes[1].set_title(f"Matriz de Confusión — Test {TEST_YEAR}")
    axes[1].set_ylabel("Real")
    axes[1].set_xlabel("Predicción")

    # 3. Curva ROC
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc_val     = auc(fpr, tpr)
    axes[2].plot(fpr, tpr, color="#3498db", linewidth=2,
                 label=f"Gradient Boosting (AUC={auc_val:.3f})")
    axes[2].plot([0,1],[0,1], "k--", alpha=0.5, label="Azar")
    axes[2].set_xlabel("Tasa Falsos Positivos")
    axes[2].set_ylabel("Tasa Verdaderos Positivos")
    axes[2].set_title(f"Curva ROC — Test {TEST_YEAR}")
    axes[2].legend(fontsize=9)
    axes[2].grid(True, alpha=0.3)

    plt.suptitle("WorldCup2026-ML — Evaluación del Modelo", fontsize=13)
    plt.tight_layout()
    plt.savefig(plots_dir / "model_evaluation.png",
                bbox_inches="tight", dpi=120)
    plt.close()
    print(f"  ✅ Gráfica guardada: outputs/plots/model_evaluation.png")


# ===========================================================
# MAIN
# ===========================================================

def main():
    print("\n" + "="*55)
    print("  WorldCup2026-ML — Entrenamiento del Modelo")
    print("="*55)

    paths = get_paths()

    # Paso 1: cargar dataset
    df = load_dataset()

    # Paso 2: walk-forward validation
    wf_metrics = walk_forward_validation(df)

    # Paso 3: entrenar modelo final
    modelo, scaler, final_metrics, y_test, y_prob, y_pred = train_final_model(df)

    # Paso 4: importancia de features
    show_feature_importance(modelo, paths)

    # Paso 5: guardar modelo
    save_model(modelo, scaler, wf_metrics, final_metrics, paths)

    # Paso 6: gráficas
    save_evaluation_plots(y_test, y_prob, y_pred, wf_metrics, paths)

    print("\n" + "="*55)
    print("  ENTRENAMIENTO COMPLETADO")
    print(f"  Walk-Forward AUC:  {wf_metrics['mean_auc']:.3f} "
          f"± {wf_metrics['std_auc']:.3f}")
    print(f"  Walk-Forward F1:   {wf_metrics['mean_f1']:.3f} "
          f"± {wf_metrics['std_f1']:.3f}")
    print(f"  Test {TEST_YEAR} AUC:   {final_metrics['auc']:.3f}")
    print(f"  Test {TEST_YEAR} F1:    {final_metrics['f1_macro']:.3f}")
    print("  Siguiente paso: python 03_simulate.py")
    print("="*55 + "\n")


if __name__ == "__main__":
    main()
