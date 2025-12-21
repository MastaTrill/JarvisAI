"""
JarvisAI Monitoring & Logging
Prometheus metrics and structured logging
"""

import logging
from pythonjsonlogger import jsonlogger
from prometheus_client import Counter, Gauge, Histogram, Summary, start_http_server
import threading
import time

# Logging configuration
logger = logging.getLogger("jarvisai")
logHandler = logging.StreamHandler()
formatter = jsonlogger.JsonFormatter()
logHandler.setFormatter(formatter)
logger.addHandler(logHandler)
logger.setLevel(logging.INFO)

# Prometheus metrics
QUANTUM_OPS_TOTAL = Counter('quantum_operations_total', 'Total quantum operations performed')
MODEL_RUNS_TOTAL = Counter('model_runs_total', 'Total model training/inference runs')
API_REQUESTS_TOTAL = Counter('api_requests_total', 'Total API requests')
ERRORS_TOTAL = Counter('errors_total', 'Total errors encountered')

CPU_USAGE = Gauge('cpu_usage_percent', 'CPU usage percent')
MEMORY_USAGE = Gauge('memory_usage_mb', 'Memory usage in MB')
DISK_USAGE = Gauge('disk_usage_percent', 'Disk usage percent')

RESPONSE_TIME = Histogram('api_response_time_seconds', 'API response time in seconds')

# Example: custom summary for quantum operation durations
QUANTUM_OP_DURATION = Summary('quantum_op_duration_seconds', 'Duration of quantum operations')


def start_metrics_server(port: int = 8001):
    """Start Prometheus metrics server in background thread"""
    def run():
        start_http_server(port)
        logger.info(f"Prometheus metrics server started on port {port}")
        while True:
            time.sleep(60)
    thread = threading.Thread(target=run, daemon=True)
    thread.start()


def log_event(event: str, **kwargs):
    """Log structured event"""
    logger.info(event, extra=kwargs)


def log_error(error: str, **kwargs):
    """Log structured error"""
    logger.error(error, extra=kwargs)
    ERRORS_TOTAL.inc()


def record_quantum_op(duration: float):
    """Record quantum operation metric"""
    QUANTUM_OPS_TOTAL.inc()
    QUANTUM_OP_DURATION.observe(duration)


def record_model_run():
    """Record model run metric"""
    MODEL_RUNS_TOTAL.inc()


def record_api_request(duration: float):
    """Record API request metric"""
    API_REQUESTS_TOTAL.inc()
    RESPONSE_TIME.observe(duration)


def update_resource_usage(cpu: float, memory: float, disk: float):
    """Update resource usage gauges"""
    CPU_USAGE.set(cpu)
    MEMORY_USAGE.set(memory)
    DISK_USAGE.set(disk)


if __name__ == "__main__":
    print("📈 Starting JarvisAI Monitoring...")
    start_metrics_server(8001)
    log_event("System startup", phase="monitoring", status="ok")
    for i in range(5):
        record_quantum_op(duration=0.01 * (i+1))
        record_model_run()
        record_api_request(duration=0.2 + 0.05 * i)
        update_resource_usage(cpu=10+i, memory=512+i*10, disk=50+i)
        time.sleep(1)
    log_error("Test error event", code=500, detail="Example error")
    print("✅ Monitoring test complete. Metrics available at :8001")
