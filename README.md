
# MNIST on K8s (Apple Silicon)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)![Kubernetes](https://img.shields.io/badge/kubernetes-%23326CE5.svg?style=for-the-badge&logo=kubernetes&logoColor=white)![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)![Prometheus](https://img.shields.io/badge/prometheus-%23E6522C.svg?style=for-the-badge&logo=Prometheus&logoColor=white)![Grafana](https://img.shields.io/badge/Grafana-F2F4F9?style=for-the-badge&logo=grafana&logoColor=orange&labelColor=F2F4F9)![Helm](https://img.shields.io/badge/Helm-0F4C73?style=for-the-badge&logo=helm&logoColor)![Redis](https://img.shields.io/badge/redis-%23DD0031.svg?style=for-the-badge&logo=redis&logoColor)

End-to-end, production-style ML system with simple MNIST modeling. Built to run locally on a MacBook (Apple Silicon) using open-source tools.

## Components
- **Data & Features**: Torchvision Datasets -> Polars feature pipeline (normalized image + per-image stats + histogram)
- **Feature Store**: Feast (offline Parquet, online Redis)
- **Training**: PyTorch Lightning CNN (CPU/MPS)
- **Serving**: FastAPI with Prometheus metrics; can fetch features from Feast or compute on the fly
- **Orchestration**: Makefile + simple scripts (add Prefect later?)
- **Observability**: Prometheus/Grafana (via Helm), `/metrics` endpoint
- **Kubernetes**: Minikube local cluster; Helm chart for API, easy to deploy to Cloud
- **CI/CD**: GitHub Actions workflow (build/test/images/deploy)
- **Drift**: Simple nightly CronJob computing histogram drift (KL divergence)

## Quickstart

### 0) Prereqs
- macOS + Apple Silicon (arm64)
- Docker Desktop **or** colima **or** another way to run containers
- kubectl, helm, minikube, python 3.11

### 1) Create and bootstrap cluster
```bash
make bootstrap
```
This installs Redis (online store), MinIO (optional), Prometheus, Grafana. Check if the cluster is running by doing

### 2) Generate features (offline Parquet) and apply Feast
```bash
make features
make feast
```

### 3) Train model
```bash
make train
```
The artifact is written to `artifacts/model.pt`.

### 4) Build & deploy API to Kubernetes
```bash
make deploy
```
Then port-forward and test:
```bash
kubectl -n ml port-forward svc/mnist-api 8080:80 &
curl -s localhost:8080/healthz
```

### 5) Metrics & Drift
- Prometheus scrapes `/metrics`; Grafana dashboards included (basic).
- Drift job (CronJob) compares recent inferences against reference features.

## Notes
- Everything is ARM64-friendly. Images use `python:3.13-slim`.
- Feast is configured to read Parquet from `./data/features.parquet` for local dev.
- For simplicity, the API image bundles the model artifact at build time.

## Roadmap (stretch)
- Deploy to VPS, host on a website
- Interactive inputs
- Swap to MLflow registry
- KServe or Ray Serve
- Evidently reports
- Argo Rollouts
