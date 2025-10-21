FROM ghcr.io/mlflow/mlflow:v3.5.0

RUN pip install boto3 botocore psycopg2-binary

# Optional: Set entrypoint or copy custom scripts if needed
CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000"]
