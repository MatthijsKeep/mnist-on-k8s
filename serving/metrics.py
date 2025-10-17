
from prometheus_client import Histogram
REQUEST_TIME = Histogram('prediction_latency_seconds', 'Prediction latency')
