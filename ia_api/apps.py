from django.apps import AppConfig

class IaApiConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'ia_api'

    def ready(self):
        # ✅ Appelé automatiquement au démarrage
        try:
            from ia_backend.services.chat_memory import create_table
            create_table()
        except Exception as e:
            import logging
            logging.warning(f"Erreur init DB chat_memory : {e}")
