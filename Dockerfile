FROM nvidia/cuda:11.8.0-base-ubuntu22.04

RUN apt-get update && apt-get install -y \
    wget \
    python3-pip \
    python3-dev \
    git

# Create and switch to app directory for application code
WORKDIR /app

COPY requirements.txt .
RUN pip3 install -r requirements.txt

# Download ESM and SAE weights
RUN mkdir -p weights
RUN wget -P weights https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t33_650M_UR50D.pt
RUN gdown https://drive.google.com/uc?id=1LtDUfcWQEwPuTdd127HEb_A2jQdRyKrU -O weights/esm2_plm1280_l24_sae4096_100Kseqs.pt

COPY handler.py .

EXPOSE 8000

CMD ["/bin/bash", "-c", "python3 handler.py"]
