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

DOS MODELOS EN ESTE SCRIPT:

    MODELO DE EVALUACIÓN:
        Entrena con datos < 2022, evalúa en 2022.
        Sirve para saber si el modelo es confiable antes
        de usarlo en producción. Si las métricas son buenas,
        procedemos a construir el modelo de producción.
        Este modelo NO se guarda en .pkl.

    MODELO DE PRODUCCIÓN:
        Entrena con datos <= 2022 (incluye 2022).
        Este es el que se guarda en .pkl y se usa para
        predecir el Mundial 2026.
        ¿Por qué incluir 2022? Porque más datos = mejor modelo.
        Ya sabemos que funciona (lo validamos arriba), entonces
        para predecir 2026 le damos toda la información disponible
        incluyendo lo que pasó en Qatar 2022.

        Ejemplo concreto: Morocco llegó a semis en 2022.
        Si no incluimos 2022, el modelo no sabe eso y puede
        subestimar a Morocco. Con 2022 incluido, su historial
        ya refleja que es un equipo capaz de llegar lejos.

TARGET:
    gana_partido = 1 si el equipo ganó ese partido
    gana_partido = 0 si perdió o empató

ENTRADA:
    data/processed/model_dataset.csv → generado por 01_etl.py

SALIDA:
    models/match_predictor.pkl → modelo de PRODUCCIÓN (<= 2022)
    models/scaler.pkl          → escalador del modelo de producción
    models/features.pkl        → lista de features usadas
    models/metrics.json        → métricas del modelo de evaluación

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

GB_PARAMS = dict(
    n_estimators=200,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.8,
    random_state=42
)

WALK_FORWARD_YEARS = [2006, 2010, 2014, 2018, 2022]
EVAL_YEAR = 2022


# ===========================================================
# PASO 1: CARGAR DATASET
# ===========================================================

def load_dataset() -> pd.DataFrame:
    """
    Carga el dataset generado por 01_etl.py.

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
        1. Train: todos los datos ANTERIORES a ese año
        2. Test:  solo los datos de ese año
        3. Entrenar modelo en train, evaluar en test
        4. Guardar métricas

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

        if train_mask.sum() < 50 or test_mask.sum() == 0:
            print(f"  {year_test:>6} → datos insuficientes, omitiendo")
            continue

        X_tr = df.loc[train_mask, MODEL_FEATURES].fillna(0)
        X_te = df.loc[test_mask,  MODEL_FEATURES].fillna(0)
        y_tr = df.loc[train_mask, TARGET]
        y_te = df.loc[test_mask,  TARGET]

        # fit SOLO en train — nunca en test
        scaler_wf = StandardScaler()
        X_tr_sc   = scaler_wf.fit_transform(X_tr)
        X_te_sc   = scaler_wf.transform(X_te)

        modelo_wf = GradientBoostingClassifier(**GB_PARAMS)
        modelo_wf.fit(X_tr_sc, y_tr)

        y_prob = modelo_wf.predict_proba(X_te_sc)[:, 1]
        y_pred = modelo_wf.predict(X_te_sc)

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
        "by_year":  results,
        "mean_auc": float(wf_df["auc"].mean()),
        "std_auc":  float(wf_df["auc"].std()),
        "mean_f1":  float(wf_df["f1_macro"].mean()),
        "std_f1":   float(wf_df["f1_macro"].std()),
        "mean_acc": float(wf_df["accuracy"].mean()),
    }


# ===========================================================
# PASO 3: MODELO DE EVALUACIÓN (train < 2022, test = 2022)
# ===========================================================

def train_evaluation_model(df: pd.DataFrame):
    """
    Entrena el modelo de EVALUACIÓN para reportar métricas.

    Entrena con datos < 2022 y evalúa en 2022.
    Responde: ¿funciona el modelo en datos reales?
    Este modelo NO se guarda en .pkl.

    RETORNA:
        metrics_eval, y_test, y_prob, y_pred
    """
    print("\n" + "="*55)
    print(f"  PASO 3: Modelo de evaluación")
    print(f"          Train: datos < {EVAL_YEAR}")
    print(f"          Test:  datos = {EVAL_YEAR}")
    print("="*55)

    train_mask = df["year"] < EVAL_YEAR
    test_mask  = df["year"] == EVAL_YEAR

    X_train = df.loc[train_mask, MODEL_FEATURES].fillna(0)
    X_test  = df.loc[test_mask,  MODEL_FEATURES].fillna(0)
    y_train = df.loc[train_mask, TARGET]
    y_test  = df.loc[test_mask,  TARGET]

    print(f"\n  Train: {len(X_train)} filas (mundiales hasta {EVAL_YEAR-1})")
    print(f"  Test:  {len(X_test)} filas  (mundial {EVAL_YEAR})")

    scaler_eval = StandardScaler()
    X_train_sc  = scaler_eval.fit_transform(X_train)
    X_test_sc   = scaler_eval.transform(X_test)

    modelo_eval = GradientBoostingClassifier(**GB_PARAMS)
    modelo_eval.fit(X_train_sc, y_train)

    y_prob = modelo_eval.predict_proba(X_test_sc)[:, 1]
    y_pred = modelo_eval.predict(X_test_sc)

    auc_val = roc_auc_score(y_test, y_prob)
    f1_val  = f1_score(y_test, y_pred, average="macro", zero_division=0)
    acc_val = accuracy_score(y_test, y_pred)

    print(f"\n  Métricas en test {EVAL_YEAR}:")
    print(f"    AUC:      {auc_val:.3f}")
    print(f"    F1-macro: {f1_val:.3f}")
    print(f"    Accuracy: {acc_val:.3f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['No ganó','Ganó'], zero_division=0)}")

    if auc_val >= 0.75:
        print(f"  ✅ AUC={auc_val:.3f} — modelo confiable para producción")
    elif auc_val >= 0.65:
        print(f"  ⚠️  AUC={auc_val:.3f} — modelo aceptable")
    else:
        print(f"  ❌ AUC={auc_val:.3f} — modelo débil, revisar features")

    metrics_eval = {
        "eval_year":  EVAL_YEAR,
        "train_size": int(len(X_train)),
        "test_size":  int(len(X_test)),
        "auc":        float(auc_val),
        "f1_macro":   float(f1_val),
        "accuracy":   float(acc_val),
    }

    return metrics_eval, y_test, y_prob, y_pred


# ===========================================================
# PASO 4: MODELO DE PRODUCCIÓN (train <= 2022)
# ===========================================================

def train_production_model(df: pd.DataFrame):
    """
    Entrena el modelo de PRODUCCIÓN para predecir el Mundial 2026.

    Incluye los datos de 2022 porque:
    - El modelo de evaluación ya confirmó que el enfoque funciona
    - Más datos históricos = mejor generalización
    - 2022 es el mundial más reciente y el más relevante para 2026

    RETORNA:
        modelo_prod, scaler_prod
    """
    print("\n" + "="*55)
    print(f"  PASO 4: Modelo de producción")
    print(f"          Train: datos <= {EVAL_YEAR} (incluye {EVAL_YEAR})")
    print(f"          Este modelo predice el Mundial 2026")
    print("="*55)

    train_mask = df["year"] <= EVAL_YEAR

    X_train = df.loc[train_mask, MODEL_FEATURES].fillna(0)
    y_train = df.loc[train_mask, TARGET]

    print(f"\n  Train: {len(X_train)} filas")
    print(f"  Años:  {sorted(df.loc[train_mask, 'year'].unique())}")

    scaler_prod = StandardScaler()
    X_train_sc  = scaler_prod.fit_transform(X_train)

    modelo_prod = GradientBoostingClassifier(**GB_PARAMS)
    modelo_prod.fit(X_train_sc, y_train)

    print(f"\n  ✅ Modelo de producción listo")
    print(f"     Entrenado con {len(X_train)} partidos (1930-{EVAL_YEAR})")

    return modelo_prod, scaler_prod


# ===========================================================
# PASO 5: IMPORTANCIA DE FEATURES
# ===========================================================

def show_feature_importance(modelo, paths: dict) -> None:
    """
    Muestra y guarda la importancia de cada feature del
    modelo de producción.
    """
    print("\n" + "="*55)
    print("  PASO 5: Importancia de features")
    print("="*55)

    imp_df = pd.DataFrame({
        "Feature":     MODEL_FEATURES,
        "Importancia": modelo.feature_importances_
    }).sort_values("Importancia", ascending=False)

    print(f"\n  {'Feature':<25} {'Importancia':>12}  Barra")
    print("  " + "-"*55)
    for _, row in imp_df.iterrows():
        barra = "█" * int(row["Importancia"] * 100)
        print(f"  {row['Feature']:<25} {row['Importancia']:>12.4f}  {barra}")

    plots_dir = paths["outputs"] / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 7))
    imp_sorted = imp_df.sort_values("Importancia", ascending=True)
    plt.barh(imp_sorted["Feature"], imp_sorted["Importancia"],
             color="#3498db")
    plt.xlabel("Importancia")
    plt.title(f"Importancia de Features — Modelo Producción (<= {EVAL_YEAR})")
    plt.tight_layout()
    plt.savefig(plots_dir / "feature_importance.png",
                bbox_inches="tight", dpi=120)
    plt.close()
    print(f"\n  ✅ outputs/plots/feature_importance.png")


# ===========================================================
# PASO 6: GUARDAR MODELO Y MÉTRICAS
# ===========================================================

def save_model(modelo_prod, scaler_prod, wf_metrics, eval_metrics, paths):
    """
    Guarda el modelo de producción y las métricas.

    Lo que se guarda:
        match_predictor.pkl → modelo producción (entrenado <= 2022)
        scaler.pkl          → escalador correspondiente
        features.pkl        → lista de features en orden correcto
        metrics.json        → métricas del modelo de evaluación
    """
    print("\n" + "="*55)
    print("  PASO 6: Guardando modelo de producción")
    print("="*55)

    models_dir = paths["models"]
    models_dir.mkdir(parents=True, exist_ok=True)

    with open(models_dir / "match_predictor.pkl", "wb") as f:
        pickle.dump(modelo_prod, f)
    print(f"\n  ✅ models/match_predictor.pkl  (producción <= {EVAL_YEAR})")

    with open(models_dir / "scaler.pkl", "wb") as f:
        pickle.dump(scaler_prod, f)
    print(f"  ✅ models/scaler.pkl")

    with open(models_dir / "features.pkl", "wb") as f:
        pickle.dump(MODEL_FEATURES, f)
    print(f"  ✅ models/features.pkl")

    all_metrics = {
        "nota": (
            f"Métricas del modelo de EVALUACIÓN (train < {EVAL_YEAR}, "
            f"test = {EVAL_YEAR}). El .pkl guardado es el modelo de "
            f"PRODUCCIÓN (train <= {EVAL_YEAR})."
        ),
        "model":        "GradientBoostingClassifier",
        "parameters":   GB_PARAMS,
        "features":     MODEL_FEATURES,
        "target":       TARGET,
        "walk_forward": wf_metrics,
        "evaluation":   eval_metrics,
    }
    with open(models_dir / "metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"  ✅ models/metrics.json")


# ===========================================================
# PASO 7: GRÁFICAS
# ===========================================================

def save_evaluation_plots(y_test, y_prob, y_pred, wf_metrics, paths):
    """Genera las gráficas de evaluación del modelo."""
    plots_dir = paths["outputs"] / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Walk-Forward
    wf_years = [r["year"]     for r in wf_metrics["by_year"]]
    wf_aucs  = [r["auc"]      for r in wf_metrics["by_year"]]
    wf_f1s   = [r["f1_macro"] for r in wf_metrics["by_year"]]

    axes[0].plot(wf_years, wf_aucs, "o-", label="AUC",
                 color="#3498db", linewidth=2, markersize=8)
    axes[0].plot(wf_years, wf_f1s,  "s-", label="F1-macro",
                 color="#e74c3c", linewidth=2, markersize=8)
    axes[0].axhline(0.5, color="gray", linestyle="--", alpha=0.5)
    axes[0].set_title("Walk-Forward Validation")
    axes[0].set_xlabel("Año de prueba")
    axes[0].set_ylabel("Score")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(0.3, 1.0)

    # Confusión
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[1],
                xticklabels=["No ganó","Ganó"],
                yticklabels=["No ganó","Ganó"])
    axes[1].set_title(f"Confusión — Test {EVAL_YEAR}")
    axes[1].set_ylabel("Real")
    axes[1].set_xlabel("Predicción")

    # ROC
    from sklearn.metrics import roc_curve, auc as sk_auc
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc_val     = sk_auc(fpr, tpr)
    axes[2].plot(fpr, tpr, color="#3498db", linewidth=2,
                 label=f"GB (AUC={auc_val:.3f})")
    axes[2].plot([0,1],[0,1],"k--",alpha=0.5)
    axes[2].set_xlabel("Falsos Positivos")
    axes[2].set_ylabel("Verdaderos Positivos")
    axes[2].set_title(f"Curva ROC — Test {EVAL_YEAR}")
    axes[2].legend(fontsize=9)
    axes[2].grid(True, alpha=0.3)

    plt.suptitle(
        f"WorldCup2026-ML — Evaluación del Modelo\n"
        f"Modelo evaluación: train < {EVAL_YEAR}, test = {EVAL_YEAR}  |  "
        f"Modelo producción guardado: train <= {EVAL_YEAR}",
        fontsize=10
    )
    plt.tight_layout()
    plt.savefig(plots_dir / "model_evaluation.png",
                bbox_inches="tight", dpi=120)
    plt.close()
    print(f"  ✅ outputs/plots/model_evaluation.png")


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

    # Paso 3: modelo de evaluación (train < 2022, test = 2022)
    eval_metrics, y_test, y_prob, y_pred = train_evaluation_model(df)

    # Paso 4: modelo de producción (train <= 2022)
    modelo_prod, scaler_prod = train_production_model(df)

    # Paso 5: importancia de features
    show_feature_importance(modelo_prod, paths)

    # Paso 6: guardar modelo de producción
    save_model(modelo_prod, scaler_prod, wf_metrics, eval_metrics, paths)

    # Paso 7: gráficas
    save_evaluation_plots(y_test, y_prob, y_pred, wf_metrics, paths)

    print("\n" + "="*55)
    print("  ENTRENAMIENTO COMPLETADO")
    print(f"\n  Evaluación (train < {EVAL_YEAR}, test = {EVAL_YEAR}):")
    print(f"    Walk-Forward AUC: {wf_metrics['mean_auc']:.3f} "
          f"± {wf_metrics['std_auc']:.3f}")
    print(f"    Walk-Forward F1:  {wf_metrics['mean_f1']:.3f} "
          f"± {wf_metrics['std_f1']:.3f}")
    print(f"    Test {EVAL_YEAR} AUC:  {eval_metrics['auc']:.3f}")
    print(f"    Test {EVAL_YEAR} F1:   {eval_metrics['f1_macro']:.3f}")
    print(f"\n  Producción (train <= {EVAL_YEAR}):")
    print(f"    ✅ Guardado en models/match_predictor.pkl")
    print(f"    Listo para predecir el Mundial 2026")
    print(f"\n  Siguiente paso: python 03_simulate.py")
    print("="*55 + "\n")


if __name__ == "__main__":
    main()
