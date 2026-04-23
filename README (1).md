# 🌄 Scenic Vibe Detector — Streamlit + Docker Deployment Guide

## Project Structure

```
scenic_vibe_app/
├── app.py                    ← Streamlit application
├── requirements.txt          ← Python dependencies
├── Dockerfile                ← Container definition
├── docker-compose.yml        ← One-command deployment
├── .streamlit/
│   └── config.toml           ← Streamlit theme & server config
└── models/                   ← ⚠️  YOU MUST CREATE THIS (see Step 1)
    ├── vibe_classifier_best.h5
    ├── quote_generator_best.h5
    ├── tokenizer.pkl
    ├── label_map.json
    └── vibe_meta.json
```

---

## Step 1 — Prepare Your Model Files

Unzip `deploy_models.zip` and copy **these 5 files** into the `models/` folder:

```bash
unzip deploy_models.zip -d extracted/
mkdir -p models/

cp extracted/vibe_classifier_best.h5   models/
cp extracted/quote_generator_best.h5   models/
cp extracted/tokenizer.pkl             models/
cp extracted/label_map.json            models/
cp extracted/vibe_meta.json            models/
```

> The `vibe_classifier_saved/` folder and `.tflite` / `.keras` files are
> **not** needed for this deployment (the app uses the `.h5` files).

---

## Step 2 — Run Locally (without Docker)

Good for a quick test before containerising.

```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Launch
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

---

## Step 3 — Build & Run with Docker

### Option A — Docker Compose (recommended)

```bash
# Build the image and start the container
docker compose up --build

# Run in the background
docker compose up --build -d

# Stop
docker compose down
```

### Option B — Plain Docker commands

```bash
# Build
docker build -t scenic-vibe-detector:latest .

# Run
docker run -d \
  --name scenic_vibe_app \
  -p 8501:8501 \
  --restart unless-stopped \
  scenic-vibe-detector:latest
```

Open **http://localhost:8501** in your browser.

---

## Step 4 — Verify the Container is Running

```bash
docker ps                              # should show scenic_vibe_app

docker logs scenic_vibe_app            # view startup logs

# Health check
curl http://localhost:8501/_stcore/health
```

---

## Step 5 — Deploy to a Cloud VM (optional)

The same Docker commands work on any Ubuntu/Debian VPS (AWS EC2, GCP Compute Engine, DigitalOcean Droplet, etc.).

```bash
# 1. SSH into your VM
# 2. Install Docker
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER && newgrp docker

# 3. Copy project files to VM (from your local machine)
scp -r scenic_vibe_app/ user@YOUR_VM_IP:~/

# 4. SSH back in and launch
ssh user@YOUR_VM_IP
cd scenic_vibe_app
docker compose up --build -d

# 5. Open port 8501 in your cloud firewall / security group
```

Your app will be available at **http://YOUR_VM_IP:8501**.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: tensorflow` | Ensure `requirements.txt` is present and `docker build` ran successfully |
| `FileNotFoundError: models/...` | Make sure all 5 model files are inside the `models/` folder before building |
| Port 8501 already in use | Change the host port: `-p 8502:8501` in docker-compose.yml |
| Slow first inference | TF loads lazily; the first prediction after startup takes ~5–10 s |
| Out of memory | TF by default grabs all GPU RAM; set `TF_FORCE_GPU_ALLOW_GROWTH=true` as an env var |

---

## App Features

- 📸 Upload any landscape image (JPG / PNG / WEBP, up to 10 MB)
- 🎨 Classifies into **8 vibes** using MobileNetV2 (Model 1)
- ✍️ Generates a contextual quote using the sequence model (Model 2)
- 📊 Shows confidence scores for all 8 vibes
- 🎲 Re-roll button for a different quote
- 🌑 Dark-mode UI themed around the vibe palette
