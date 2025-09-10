# webapp/rpy2_init.py
import threading
import warnings

_initialized = False

def warmup_r():
    """Inicializa rpy2/R una sola vez desde el hilo principal, sin activar conversores globales."""
    global _initialized
    if _initialized:
        return
    if threading.current_thread() is not threading.main_thread():
        return

    # En algunos entornos convierten warnings en errores: los silenciamos durante el warm-up.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=Warning)

        # Forzar carga/arranque de R de forma compatible con versiones
        try:
            import rpy2.rinterface as ri
            if hasattr(ri, "initr"):
                ri.initr()  # en versiones antiguas
        except Exception:
            pass

        # Importar robjects ya inicializa R en versiones modernas
        from rpy2 import robjects as ro
        ro.r("invisible(1+1)")

    _initialized = True
