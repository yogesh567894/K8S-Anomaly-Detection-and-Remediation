# Kubernetes Anomaly Visualization Dashboard

A Streamlit-based dashboard for visualizing Kubernetes cluster state, detecting anomalies, and performing remediation actions.

## Features

- **Cluster Topology Visualization**: Interactive graph visualization of your Kubernetes cluster showing nodes, pods and their relationships
- **Anomaly Detection**: Real-time detection and highlighting of anomalous pods
- **Time-Series Metrics**: Visual representation of pod metrics over time
- **AI-Powered Insights**: Natural language explanations of detected anomalies powered by NVIDIA LLM
- **Interactive Remediation**: Execute remediation actions directly from the UI with approval workflow

## Installation

1. Install the required dependencies:

```bash
pip install -r requirements-dashboard.txt
```

2. Make sure your Kubernetes configuration is properly set up:
   - For local clusters: `~/.kube/config` should be configured
   - For in-cluster deployment: Service account with appropriate permissions

3. Set up the LLM API key for AI insights:

```bash
# For NVIDIA NeMo API
export NVIDIA_API_KEY="your-api-key-here"

# Or for OpenAI (if configured)
export OPENAI_API_KEY="your-api-key-here"
```

## Running the Dashboard

Launch the dashboard using the provided script:

```bash
# From the project root
python src/run_visualization_dashboard.py
```

Or directly with Streamlit:

```bash
streamlit run src/agents/k8s_visualization_dashboard.py
```

The dashboard will be accessible at `http://localhost:8501`

## Usage

### Cluster Visualization

- The main graph shows your cluster's topology
- Nodes are represented as blue circles
- Pods are represented as green circles
- Anomalous pods are highlighted in red
- Click on any node or pod to view details

### Anomaly Detection

- Detected anomalies appear in the right panel
- Each anomaly includes:
  - Anomaly type
  - Confidence score
  - Pod metrics
  - Associated events
  - Remediation options

### Remediation Actions

For each detected anomaly, you can:

1. View the AI-suggested remediation
2. Select an action from the dropdown:
   - `restart_pod`: Restart the problematic pod
   - `restart_deployment`: Restart the entire deployment
   - `increase_memory`: Increase memory allocation
   - `increase_cpu`: Increase CPU allocation
   - `scale_deployment`: Add more replicas
3. Execute the action with the "Execute Remediation" button
4. View the results of the action

### Options and Settings

In the sidebar, you can:

- Enable "Test Mode" to use sample data instead of a live cluster
- Adjust the refresh interval
- Enable auto-refresh for continuous monitoring
- Set the anomaly confidence threshold

## Architecture

The dashboard integrates with:
- The K8s Multi-Agent System for metrics collection and anomaly detection
- Kubernetes API for cluster management and remediation
- NVIDIA LLM API for AI-powered insights

## Troubleshooting

- **Dashboard fails to start**: Ensure all dependencies are installed
- **No cluster data**: Check your Kubernetes configuration
- **AI insights unavailable**: Verify your LLM API key is set
- **Remediation fails**: Check logs and ensure proper permissions

For detailed error logs, run with the debug flag:

```bash
python src/run_visualization_dashboard.py --debug
``` 