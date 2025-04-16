# Kubernetes Monitoring and Remediation System

A platform for monitoring Kubernetes clusters, detecting anomalies, generating insights, and performing remediation with approval.

## Overview

This system provides real-time Kubernetes monitoring with ML-based anomaly detection and automated remediation. It combines metrics collection, analysis, and optional LLM integration for context-aware recommendations.

## Quick Start

```bash
# Clone and set up environment
git clone https://github.com/yourusername/kubernetes-monitoring.git
cd kubernetes-monitoring
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac
pip install -r requirements.txt

# Start Kubernetes locally
minikube start
kubectl port-forward -n monitoring svc/prometheus-kube-prometheus-prometheus 9090:9090

# Run the system (all components)
python run_monitoring.py

# Or run individual components
python dataset-generator.py  # Collect metrics
python src/agents/dataset_generator_agent.py  # Process and analyze metrics
python run_remediation.py  # Handle remediation
```

## Components

- **Dataset Generator** (`dataset-generator.py`): Collects K8s pod metrics (CPU, memory, network)
- **Dataset Generator Agent** (`src/agents/dataset_generator_agent.py`): Processes metrics and detects anomalies
- **Anomaly Detection Agent** (`src/agents/anomaly_detection_agent.py`): Analyzes metrics for unusual patterns
- **Remediation Agent** (`run_remediation.py`): Executes approved corrective actions
- **Multi-Agent System** (`k8s_multi_agent_system.py`): Orchestrates all components together

## Key Features

- **Metrics Collection**: Real-time pod and node metrics via direct API or Prometheus
- **Anomaly Detection**: ML-based detection of unusual pod behavior
- **Actionable Insights**: Automatic recommendations for detected issues
- **Automated Remediation**: Execute fixes with approval (restart pods, scale deployments)
- **LLM Integration**: Enhanced analysis with natural language capabilities

## Usage Options

### Dataset Generator Agent Modes

```bash
python src/agents/dataset_generator_agent.py         # Run both dataset generator and agent
python src/agents/dataset_generator_agent.py -agent   # Run only agent (analyze metrics)
python src/agents/dataset_generator_agent.py -fetch   # Run only generator (collect metrics)
```

### Monitoring Script Modes

```bash
python run_monitoring.py                      # Run all components
python run_monitoring.py --generator-only     # Run only dataset generator
python run_monitoring.py --dataset-agent-only # Run only dataset agent
python run_monitoring.py --openai-api-key KEY # Enable LLM features
```

## Configuration

- Set options via command line arguments or environment variables
- Use YAML configuration files in the `config/` directory
- Key environment variables:
  ```bash
  export OPENAI_API_KEY=your-api-key  # For LLM features
  export KUBECONFIG=/path/to/config   # For custom K8s config
  export ANOMALY_THRESHOLD=0.8        # For detection sensitivity
  ```

## Local Kubernetes Setup

1. **Start Minikube**: `minikube start --cpus=4 --memory=8192 --driver=docker`
2. **Setup Prometheus**:
   ```bash
   helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
   helm repo update
   kubectl create namespace monitoring
   helm install prometheus prometheus-community/kube-prometheus-stack --namespace monitoring
   kubectl port-forward -n monitoring svc/prometheus-kube-prometheus-prometheus 9090:9090
   ```

## Outputs

- **pod_metrics.csv**: Raw metrics data
- **pod_insights.json**: Anomaly insights
- **k8s_monitoring.log**: System logs

## Troubleshooting

- **Connection Issues**: Check `kubectl config view` and `kubectl cluster-info`
- **Missing Metrics**: Verify Prometheus setup and port-forwarding
- **Model Errors**: Ensure model files exist in expected locations
- **Remediation Failures**: Check `k8s_remediation.log` for error details

## License

MIT 