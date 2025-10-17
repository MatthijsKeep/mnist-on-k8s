.PHONY: help setup bootstrap features feast train build deploy push drift-run destroy

# Variables
NAMESPACE ?= ml
IMAGE_NAME ?= mnist-api
IMAGE_TAG ?= local# Override for versions, e.g., make deploy IMAGE_TAG=v1.0
REGISTRY ?= yourusername# Set for cloud, e.g., your Docker Hub/ECR
LOCAL ?= true# true for Minikube local; false for cloud/VPS
MINIKUBE_PROFILE ?= mnist
CLUSTER_NAME ?= $(MINIKUBE_PROFILE)

# Full image ref
# After variables section
ifeq ($(LOCAL),true)
REPO_PATH := $(IMAGE_NAME)# e.g., mnist-api (local)
else
REPO_PATH := $(REGISTRY)/$(IMAGE_NAME)# e.g., yourusername/mnist-api (remote)
endif




# Echo value of local
print-local:
	@echo "LOCAL is set to '$(LOCAL)'"

print-repo:
	@echo "Image repository path is '$(REPO_PATH)'"


help:  ## Show this help
	@egrep -h '\s##\s' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

setup:  ## Prep cluster (Minikube start if LOCAL; skip for cloud)
ifeq ($(LOCAL),true)
	minikube start --profile $(MINIKUBE_PROFILE) --driver=qemu --cpus=4 --memory=8192 --extra-config=apiserver.service-node-port-range=1-65535 || true
	kubectl config use-context $(MINIKUBE_PROFILE) || true
	minikube addons enable ingress --profile $(MINIKUBE_PROFILE) || true
	kubectl label nodes $(MINIKUBE_PROFILE) ingress-ready=true --overwrite || true
	@echo "Minikube '$(MINIKUBE_PROFILE)' ready."
else
	@echo "Skipping Minikube setup for remote cluster; ensure kubectl/helm point to it."
endif

bootstrap: setup  ## Install deps (Redis, Prometheus) in namespace (local only)
ifeq ($(LOCAL),true)
	helm repo add bitnami https://charts.bitnami.com/bitnami || true
	helm repo add prometheus-community https://prometheus-community.github.io/helm-charts || true
	kubectl create ns $(NAMESPACE) || true
	helm upgrade --install redis bitnami/redis -n $(NAMESPACE) --set architecture=standalone
	helm upgrade --install kube-prom-stack prometheus-community/kube-prometheus-stack -n $(NAMESPACE) \
		--set grafana.enabled=true --set grafana.service.type=ClusterIP
	@echo "Bootstrap complete for '$(NAMESPACE)'."
else
	@echo "Skipping bootstrap for remote; run manually if needed."
endif

features:  ## Run feature pipeline
	uv run python -m pipelines.feature_pipeline

feast:  ## Apply and materialize Feast
	cd features/feast_repo && uv run feast apply && uv run feast materialize-incremental $$(date +%F)

train:  ## Train model
	uv run python -m pipelines.train

deploy: build
	helm upgrade --install mnist-api infra/helm/api -n $(NAMESPACE) \
		--set image.repository=$(REPO_PATH) \
		--set image.tag=$(IMAGE_TAG)

build:
ifeq ($(LOCAL),true)
	eval $$(minikube -p $(MINIKUBE_PROFILE) docker-env) && docker build -f Dockerfile.api -t $(IMAGE_NAME):$(IMAGE_TAG) .  # Note: tags local name
else
	docker build -f Dockerfile.api -t $(REPO_PATH):$(IMAGE_TAG) .
	docker push $(REPO_PATH):$(IMAGE_TAG)
endif

push:  ## Explicit push to registry (for cloud, after build)
ifeq ($(LOCAL),true)
	@echo "Push not needed for local Minikube."
else
	docker push $(REGISTRY)/$(IMAGE_NAME):$(IMAGE_TAG)
endif

drift-run:  ## Run drift job
	uv run python -m drift.drift_job

destroy:  ## Teardown (Minikube delete; Helm uninstall for both)
	helm uninstall mnist-api -n $(NAMESPACE) || true
	helm uninstall redis -n $(NAMESPACE) || true
	helm uninstall kube-prom-stack -n $(NAMESPACE) || true
ifeq ($(LOCAL),true)
	minikube delete --profile $(MINIKUBE_PROFILE) || true
	@echo "Local cluster deleted."
else
	@echo "Uninstalled releases in remote namespace '$(NAMESPACE)'; cleanup resources manually."
endif
