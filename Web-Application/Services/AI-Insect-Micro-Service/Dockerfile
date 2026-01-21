FROM python:3.13-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements. txt . 
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY Insect_Prediction_flask_api.py .
COPY gunicorn_config.py . 

# Copy model directory
COPY model/ /app/model/

# Pre-warm the YOLO model to avoid runtime downloads
# This prevents the model from downloading extra files on first run
RUN python -c "from ultralytics import YOLO; YOLO('./model/last.pt')"

# Expose port
EXPOSE 7733

# Run with gunicorn
CMD ["gunicorn", "-c", "gunicorn_config. py", "Insect_Prediction_flask_api:app"]