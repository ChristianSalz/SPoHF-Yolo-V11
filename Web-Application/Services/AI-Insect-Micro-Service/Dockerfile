FROM python:3.13-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt . 

# Install CPU-only PyTorch FIRST (before ultralytics) - NEWEST VERSIONS
RUN pip install --no-cache-dir torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cpu

# Then install other requirements
RUN pip install --no-cache-dir --disable-pip-version-check -r requirements.txt

# Explicitly install ultralytics without optional dependencies (like CUDA)
RUN pip install --no-cache-dir ultralytics==8.4.7 --no-deps

COPY Insect_Prediction_flask_api.py .  
COPY gunicorn_config.py . 

COPY model/ /app/model/

EXPOSE 7733

CMD ["gunicorn", "-c", "gunicorn_config.py", "Insect_Prediction_flask_api:app"]