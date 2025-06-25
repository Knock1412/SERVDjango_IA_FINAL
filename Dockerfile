FROM python:3.9-slim

# Dépendances système utiles pour PyMuPDF, paddleocr, etc.
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    tesseract-ocr \
    libpoppler-cpp-dev \
    poppler-utils \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Dossier de travail
WORKDIR /app

# Copie tous les fichiers du projet
COPY . /app

# Installation des packages Python
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Installation du modèle de traduction Argo
RUN python install_argos_models.py

# Port Django
EXPOSE 8000

# Commande par défaut (peut être override par docker-compose)
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
