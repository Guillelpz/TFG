from django.apps import AppConfig
import logging
import os
import threading


class WebappConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "webapp"

    _did_warmup = False  # para no ejecutar dos veces

    def ready(self):
        """
        Este método corre en el hilo principal de cada worker de Gunicorn,
        o en el proceso de desarrollo de Django.
        """
        try:
            # En modo dev, el autoreloader arranca dos procesos: 
            # solo el hijo (RUN_MAIN=true) debe inicializar R.
            if os.environ.get("RUN_MAIN") == "true" and threading.current_thread() is threading.main_thread():
                if not WebappConfig._did_warmup:
                    from .rpy2_init import warmup_r
                    warmup_r()
                    WebappConfig._did_warmup = True

            # En Gunicorn/producción, RUN_MAIN no existe, pero seguimos verificando hilo principal.
            elif threading.current_thread() is threading.main_thread():
                if not WebappConfig._did_warmup:
                    from .rpy2_init import warmup_r
                    warmup_r()
                    WebappConfig._did_warmup = True

        except Exception as e:
            logging.getLogger(__name__).warning("rpy2 warmup failed: %s", e)
