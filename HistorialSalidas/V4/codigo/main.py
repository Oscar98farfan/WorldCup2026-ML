"""
===========================================================
main.py — Pipeline completo WorldCup2026-ML
===========================================================

PROPÓSITO:
    Ejecuta los 4 scripts del proyecto en orden.
    Es el punto de entrada principal del proyecto.

ORDEN DE EJECUCIÓN:
    1. 01_etl.py       → limpia datos y construye features
    2. 02_train.py     → entrena y valida el modelo
    3. 03_simulate.py  → simula el torneo 2026 (10,000 veces)
    4. 04_report.py    → genera gráficas y reporte final

¿CUÁNDO USAR ESTE SCRIPT?
    Cuando quieras correr todo de cero:
        python main.py

¿CUÁNDO CORRER LOS SCRIPTS POR SEPARADO?
    - Solo actualizaste los grupos 2026:
        python 03_simulate.py
        python 04_report.py

    - Solo quieres regenerar los reportes:
        python 04_report.py

    - Empezar desde cero con datos nuevos:
        python main.py

CÓMO EJECUTAR:
    python main.py                    → pipeline completo
    python main.py --skip-etl        → omite el ETL
    python main.py --only-simulate   → solo simula y reporta

AUTOR: Oscar Daniel Farfan Juanias
===========================================================
"""

import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))


def run_step(step_name: str, step_func, skip: bool = False) -> bool:
    """
    Ejecuta un paso del pipeline con manejo de errores.

    PARÁMETROS:
        step_name → nombre del paso para mostrar en consola
        step_func → función a ejecutar
        skip      → si True, omite el paso

    RETORNA:
        True si el paso se ejecutó correctamente
        False si hubo un error
    """
    if skip:
        print(f"\n  ⏭️  {step_name} omitido")
        return True

    print(f"\n{'='*55}")
    print(f"  Ejecutando: {step_name}")
    print(f"{'='*55}")

    start = time.time()
    try:
        step_func()
        elapsed = time.time() - start
        print(f"\n  ✅ {step_name} completado en {elapsed:.1f}s")
        return True
    except Exception as e:
        print(f"\n  ❌ Error en {step_name}:")
        print(f"     {e}")
        return False


def main():
    # Parsear argumentos simples
    args        = sys.argv[1:]
    skip_etl    = "--skip-etl" in args
    only_sim    = "--only-simulate" in args

    if only_sim:
        skip_etl = True

    print("\n" + "="*55)
    print("  WorldCup2026-ML — Pipeline Completo")
    print("="*55)
    print(f"\n  Pasos a ejecutar:")
    print(f"    {'⏭️ ' if skip_etl else '✓ '} 01_etl.py       — ETL")
    print(f"    {'⏭️ ' if skip_etl else '✓ '} 02_train.py     — Entrenamiento")
    print(f"    ✓  03_simulate.py  — Simulación (puede tardar ~3-5 min)")
    print(f"    ✓  04_report.py    — Reportes")

    total_start = time.time()
    success     = True

    # Importar funciones main de cada script
    from importlib import import_module

    steps = [
        ("01_etl.py — ETL",              "etl_01",      skip_etl),
        ("02_train.py — Entrenamiento",  "train_02",    skip_etl),
        ("03_simulate.py — Simulación",  "simulate_03", False),
        ("04_report.py — Reportes",      "report_04",   False),
    ]

    # Importar y ejecutar cada módulo
    module_map = {
        "etl_01":      ("01_etl",      "main"),
        "train_02":    ("02_train",    "main"),
        "simulate_03": ("03_simulate", "main"),
        "report_04":   ("04_report",   "main"),
    }

    for step_name, module_key, skip in steps:
        module_name, func_name = module_map[module_key]
        try:
            mod  = import_module(module_name.replace("-", "_")
                                 .replace("01_etl",      "etl_module")
                                 .replace("02_train",    "train_module")
                                 .replace("03_simulate", "simulate_module")
                                 .replace("04_report",   "report_module"))
        except Exception:
            pass

        if not skip:
            ok = run_step(step_name, _get_main(module_name), skip)
            if not ok:
                print(f"\n  Pipeline detenido en: {step_name}")
                success = False
                break

    total_elapsed = time.time() - total_start

    print("\n" + "="*55)
    if success:
        print("  PIPELINE COMPLETADO ✅")
        print(f"  Tiempo total: {total_elapsed:.1f}s "
              f"({total_elapsed/60:.1f} min)")
        print("\n  Resultados:")
        print("    outputs/simulation_results.csv")
        print("    outputs/report_2026.txt")
        print("    outputs/plots/predictions_by_round.png")
        print("    outputs/plots/top_candidates.png")
        print("    outputs/plots/group_analysis.png")
    else:
        print("  PIPELINE FALLÓ ❌")
        print("  Revisa el error arriba y vuelve a intentar")
    print("="*55 + "\n")


def _get_main(module_name: str):
    """
    Importa y retorna la función main() de un script.
    Permite ejecutar cada script como función desde main.py.
    """
    import importlib.util
    script_path = Path(__file__).parent / f"{module_name}.py"

    spec   = importlib.util.spec_from_file_location(module_name, script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module.main


if __name__ == "__main__":
    main()
