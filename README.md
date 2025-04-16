# Kubernetes Metrics and Anomaly Detection

This project provides tools to gather Kubernetes pod metrics, detect anomalies, generate actionable insights, and perform remediation with user approval.

## Components

1. **Dataset Generator** (`dataset-generator.py`): Collects real-time metrics from Kubernetes pods
2. **Anomaly Prediction Model** (`anomaly_prediction.py`): LSTM-based model for detecting anomalies in pod metrics
3. **Dataset Generator Agent** (`dataset_generator_agent.py`): Monitors metrics and provides actionable insights
4. **Anomaly Agent** (`anomaly_agent.py`): LLM-powered agent for detailed analysis and recommendations
5. **Remediation Agent** (`remediation_agent.py`): Interactive agent that requests approval before remediating detected issues

## Setup

### Prerequisites

- Python 3.8+
- Kubernetes cluster (Minikube/local or remote)
- Required Python packages:
  ```
  pandas>=1.3.0
  numpy>=1.20.0
  kubernetes>=12.0.0
  tensorflow>=2.8.0
  scikit-learn>=1.0.0
  joblib>=1.1.0
  langchain>=0.0.267
  langgraph>=0.0.17
  ```

### Installation

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Configure Kubernetes access:
   - For Minikube: `minikube start`
   - For remote clusters: Ensure your `~/.kube/config` is properly configured

## Usage

### 1. Generate Dataset

```bash
python dataset-generator.py
```

This will collect metrics from your Kubernetes cluster and save them to `pod_metrics.csv`.

Options:
- The script is configured by default for Minikube with Prometheus at `http://localhost:8082`
- Edit the constants at the top of the script for different setups

### 2. Run Dataset Generator Agent

```bash
python dataset_generator_agent.py --input-file pod_metrics.csv --watch-interval 10
```

Options:
- `--input-file`: Path to the metrics CSV file (default: `pod_metrics.csv`)
- `--watch-interval`: Seconds between checks (default: 10)
- `--alert-threshold`: Probability threshold for anomaly alerts (default: 0.7)
- `--history-window`: Number of minutes of history to maintain (default: 60)

### 3. Run Anomaly Agent with LLM

```bash
python anomaly_agent.py
```

For LLM support, set one of the following environment variables:
- OpenAI API: `export OPENAI_API_KEY=your-api-key`
- NVIDIA API: `export NVIDIA_API_KEY=nvapi-your-key`

### 4. Run Remediation Agent

```bash
python run_remediation.py --metrics-file pod_metrics.csv --watch-interval 10 --confidence-threshold 0.7
```

Options:
- `--metrics-file`: Path to the metrics CSV file (default: `pod_metrics.csv`)
- `--watch-interval`: Seconds between checks (default: 10)
- `--confidence-threshold`: Probability threshold for anomaly alerts (default: 0.7)
- `--auto-approve`: Automatically approve remediation actions (USE WITH CAUTION)

### 5. Test Remediation Agent (without a live cluster)

```bash
python test_remediation_agent.py
```

This runs the remediation agent with mock Kubernetes clients and sample anomaly data.

## Workflow

1. **Data Collection**: The dataset generator connects to your Kubernetes cluster and collects metrics
2. **Monitoring**: The dataset generator agent watches the CSV file for new entries
3. **Analysis**: The agent processes new data and detects anomalies using the ML model
4. **Insights**: Actionable recommendations are generated based on detected issues
5. **LLM Analysis**: For deeper analysis, the anomaly agent can provide detailed explanations
6. **Remediation**: The remediation agent suggests and implements corrective actions with user approval

## Remediation Actions

The remediation agent can perform the following actions:

- **Pod Restart**: Delete and recreate problematic pods
- **Memory Increase**: Adjust memory limits for deployments experiencing OOM kills
- **Deployment Scaling**: Scale up deployments facing resource exhaustion
- **Custom Remediation**: Tailored recommendations based on anomaly type

All actions require explicit user approval before execution.

## Output

- **Console Output**: Real-time alerts, recommendations, and remediation prompts
- **CSV File**: Raw metrics data in `pod_metrics.csv`
- **JSON Insights**: Anomaly insights in `pod_insights.json`
- **Logs**: Detailed logs in `dataset_generator_agent.log` and `k8s_remediation.log`

## Models and Files

- `lstm_anomaly_model.h5`: Pre-trained LSTM model for anomaly detection
- `scaler.pkl`: Feature scaler for normalizing input data
- `anomaly_threshold.pkl`: Threshold for anomaly classification

## Troubleshooting

- **Connection Issues**: Ensure your Kubernetes context is correct and the cluster is accessible
- **Missing Metrics**: Check if Prometheus is correctly set up and port-forwarding is active
- **Model Errors**: Ensure all required model files are in the same directory
- **Remediation Failures**: Check `k8s_remediation.log` for detailed error messages

## License

MIT 