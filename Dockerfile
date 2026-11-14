FROM python:3.14-slim

WORKDIR /app

# Install OpenMP runtime required by LightGBM on slim Linux images
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY predict.py serve.py model.bin ./

EXPOSE 9696

CMD ["python", "serve.py"]