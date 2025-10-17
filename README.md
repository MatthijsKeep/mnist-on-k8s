
# MNIST on K8s (Apple Silicon friendly)

End-to-end, production-style ML system with simple MNIST modeling. Built to run locally on a MacBook (Apple Silicon) using open-source tools.

## Components
- **Data & Features**: Polars feature pipeline (per-image stats + histogram)
- **Feature Store**: Feast (offline Parquet, online Redis)
- **Training**: PyTorch Lightning logistic regression (CPU/MPS)
- **Serving**: FastAPI with Prometheus metrics; can fetch features from Feast or compute on the fly
- **Orchestration**: Makefile + simple scripts (add Prefect later if desired)
- **Observability**: Prometheus/Grafana (via Helm), `/metrics` endpoint
- **Kubernetes**: kind local cluster; Helm chart for API
- **CI/CD**: GitHub Actions workflow (build/test/images/deploy)
- **Drift**: Simple nightly CronJob computing histogram drift (KL divergence)

## Quickstart

### 0) Prereqs
- macOS + Apple Silicon (arm64)
- Docker Desktop **or** colima
- kubectl, helm, kind, python 3.11

### 1) Create and bootstrap cluster
```bash
make bootstrap
```
This installs Redis (online store), MinIO (optional), Prometheus, Grafana.

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
- Everything is ARM64-friendly. Images use `python:3.11-slim`.
- Feast is configured to read Parquet from `./data/features.parquet` for local dev.
- For simplicity, the API image bundles the model artifact at build time.

## Roadmap (stretch)
- Swap to MLflow registry; KServe or Ray Serve; Evidently reports; Argo Rollouts
