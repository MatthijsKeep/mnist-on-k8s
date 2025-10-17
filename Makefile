
.PHONY: bootstrap features feast train api-build deploy drift-run destroy minikube

NAMESPACE=ml
CLUSTER_NAME=mnist
MINIKUBE_PROFILE=$(CLUSTER_NAME)

minikube:
	minikube start --profile $(MINIKUBE_PROFILE) --driver=qemu --cpus=4 --memory=8192 --extra-config=apiserver.service-node-port-range=1-65535 || true
	kubectl config use-context $(MINIKUBE_PROFILE) || true
	minikube addons enable ingress --profile $(MINIKUBE_PROFILE) || true
	kubectl label nodes $(MINIKUBE_PROFILE) ingress-ready=true --overwrite || true
	@echo "Minikube cluster '$(MINIKUBE_PROFILE)' started with ingress enabled and node labeled."

bootstrap: minikube
	helm repo add bitnami https://charts.bitnami.com/bitnami || true
	helm repo add prometheus-community https://prometheus-community.github.io/helm-charts || true
	kubectl create ns $(NAMESPACE) || true
	helm upgrade --install redis bitnami/redis -n $(NAMESPACE) --set architecture=standalone
	helm upgrade --install kube-prom-stack prometheus-community/kube-prometheus-stack -n $(NAMESPACE) \
		--set grafana.enabled=true --set grafana.service.type=ClusterIP
	@echo "Bootstrap complete for namespace '$(NAMESPACE)'."

features:
	uv run python -m pipelines.feature_pipeline

feast:
	cd features/feast_repo && uv run feast apply && uv run feast materialize-incremental $$(date +%F)

train:
	uv run python -m pipelines.train

api-build:
	docker build -f Dockerfile.api -t mnist-api:local .

deploy: api-build
	helm upgrade --install mnist-api infra/helm/api -n $(NAMESPACE) --set image.repository=mnist-api --set image.tag=local

drift-run:
	uv run python -m drift.drift_job

destroy:
	minikube delete --profile $(MINIKUBE_PROFILE) || true
	@echo "Minikube cluster '$(MINIKUBE_PROFILE)' deleted."
