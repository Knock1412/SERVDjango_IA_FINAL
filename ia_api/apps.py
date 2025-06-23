from django.apps import AppConfig

class IaApiConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'ia_api'

    def ready(self):
        # ✅ Appelé automatiquement au démarrage
        import logging

        try:
            from ia_backend.services.chat_memory import create_table
            create_table()
        except Exception as e:
            logging.warning(f"Erreur init DB chat_memory : {e}")

        try:
            from ia_backend.services.metadata_db import init_db
            init_db()
        except Exception as e:
            logging.warning(f"Erreur init DB metadonnees : {e}")
