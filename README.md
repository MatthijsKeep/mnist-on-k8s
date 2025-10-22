
# MNIST on K8s (Apple Silicon)

[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)[![Kubernetes](https://img.shields.io/badge/kubernetes-%23326CE5.svg?style=for-the-badge&logo=kubernetes&logoColor=white)](https://kubernetes.io)[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com)[![Prometheus](https://img.shields.io/badge/prometheus-%23E6522C.svg?style=for-the-badge&logo=Prometheus&logoColor=white)](https://prometheus.io)[![Grafana](https://img.shields.io/badge/Grafana-333333?style=for-the-badge&logo=grafana&logoColor=orange&labelColor=333333)](https://grafana.com)[![Helm](https://img.shields.io/badge/Helm-0F4C73?style=for-the-badge&logo=helm&logoColor)](https://helm.sh)[![Redis](https://img.shields.io/badge/redis-%23DD0031.svg?style=for-the-badge&logo=redis&logoColor)](https://redis.io)[![Polars](https://img.shields.io/badge/Polars-0275ff?style=for-the-badge&logo=polars&logoColor)](https://pola.rs) 


![digits_gif](https://github.com/user-attachments/assets/74e07858-d91b-4cf5-b3dc-8ac17e6c2d34)


End-to-end, production-emulating ML system. Uses MNIST as example, but any dataset/model can be used with PyTorch. Built to run locally on Kubernetes on a MacBook (Apple Silicon).  My first attempt at getting some experience with Kubernetes.

## Components
- **Data & Features**: Torchvision Datasets (MNIST) -> Polars feature pipeline (normalized image + per-image stats + histogram)
- **Feature Store**: Feast (offline uses Parquet, online Redis on Kubernetes)
- **Training**: PyTorch Lightning CNN (on CPU or MPS for acceleration)
- **Serving**: FastAPI, with Prometheus metrics; can fetch features from Feast or compute on the fly if you send a file (image in this case)
- **Orchestration**: Makefile for easy build and deploy(add Prefect later?)
- **Observability**: Prometheus/Grafana (via Helm), `/metrics` endpoint, to be built out (latency, amount of calls)
- **Kubernetes**: Minikube local cluster; Helm chart for API, to be expanded to more resources, easy to deploy to Cloud
- **CI/CD**: GitHub Actions workflow (build/test/images/deploy), not set up yet
- **Drift**: CronJob computing histogram drift (KL divergence between distributions)

TODO: Insert architecture diagram

## Quickstart

### 0) Prereqs
- macOS + Apple Silicon (arm64) or any other arm64 chip. It is possible to run with another but you will have to adjust your Kubernetes setup. If you already have a Kubernetes cluster running you will be fine.
- Docker Desktop **or** colima **or** another way to run containers, I use qemu
- kubectl, helm, minikube, python, uv (package management)

### 1) Create and bootstrap cluster
```bash
make bootstrap
```
This creates the cluster and installs Redis (online store), MinIO (optional), Prometheus and Grafana. Check if the cluster is running by doing `kubectl get nodes`. You should see a cluster with the name you chose for the namespace (default is `ml`)

### 2) Generate features (offline Parquet) and apply Feast
```bash
make features
make feast
```
Create the feature store, and push to Feast, which we use for versioning. 

### 3) Train model
```bash
make train
```
The artifact is written to `artifacts/model.pt`. This is later picked up by the API for serving. You can look at the experiment in MLFlow at `localhost:5000`.

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
- For simplicity, the API image bundles the model artifact at build time. You can argue that it might be better to do this from a model registry, for versioning. (see roadmap)

## Todo
- Fix cicd
- Create GUI where you can draw digits and send to model

## Roadmap (stretch)
- Deploy to VPS, host on a website
- Interactive inputs (draw box)
- Swap to MLflow registry
- KServe or Ray Serve
- Evidently reports
- Argo Rollouts


<br> 

[![hoppscotch](https://img.shields.io/badge/Tested_on-hoppscotch-202124?logo=hoppscotch&style=for-the-badge&labelColor=64d58e&logoColor=ffffff)](https://hoppscotch.io)