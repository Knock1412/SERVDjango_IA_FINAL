import argostranslate.package

model_path = "/Users/imptc03/Desktop/Projet Stage IMPERIATEC-Adrien/SERVDjango_IA_FINAL-master/translate-en_fr-1_9.argosmodel"

argostranslate.package.install_from_path(model_path)

print("Modèle installé avec succès.")
