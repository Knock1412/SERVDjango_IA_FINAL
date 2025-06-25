# install_argos_models.py
import argostranslate.package

model_path = "translate-en_fr-1_9.argosmodel"
argostranslate.package.install_from_path(model_path)
print("✅ Modèle Argos installé avec succès.")
