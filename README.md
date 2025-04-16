# Kubernetes Monitoring and Remediation System: Architecture Overview

This document provides a comprehensive overview of the architecture, code organization, and implementation details of the Kubernetes Monitoring and Remediation System.

## Table of Contents

- [System Architecture Overview](#system-architecture-overview)
- [Component Details](#component-details)
  - [Data Collection Layer](#data-collection-layer)
  - [Processing Layer](#processing-layer)
  - [Analysis Layer](#analysis-layer)
  - [Remediation Layer](#remediation-layer)
  - [Orchestration Layer](#orchestration-layer)
- [Data Flow](#data-flow)
- [Implementation Details](#implementation-details)
- [Directory Structure](#directory-structure)
- [Integration Points](#integration-points)
- [Configuration System](#configuration-system)
- [Extension Points](#extension-points)
- [Deployment Considerations](#deployment-considerations)
- [Getting Started](#getting-started)

## System Architecture Overview

The system follows a layered architecture designed to separate concerns and provide modularity. Each layer has specific responsibilities and interfaces with adjacent layers:

```
┌─────────────────────────────────────────────────────────────────┐
│                     Kubernetes Cluster                          │
│  ┌───────────┐    ┌───────────┐    ┌───────────┐  ┌──────────┐  │
│  │ Pods      │    │ Nodes     │    │ Services  │  │Prometheus│  │
│  └───────────┘    └───────────┘    └───────────┘  └──────────┘  │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Data Collection Layer                        │
│  ┌────────────────────┐        ┌─────────────────────────────┐  │
│  │ dataset-generator.py│        │run_agent_with_custom_metrics│  │
│  └────────────────────┘        └─────────────────────────────┘  │
│                               │                                  │
│                               ▼                                  │
│                      ┌───────────────┐                          │
│                      │pod_metrics.csv │                          │
│                      └───────────────┘                          │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Processing Layer                            │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                 Dataset Generator Agent                     │ │
│  │            (src/agents/dataset_generator_agent.py)          │ │
│  └────────────────────────────────────────────────────────────┘ │
│                               │                                  │
│                               ▼                                  │
│  ┌────────────────────┐      │     ┌───────────────────────┐    │
│  │ Anomaly Prediction │◄─────┴────►│    pod_insights.json  │    │
│  │        Model       │            └───────────────────────┘    │
│  └────────────────────┘                                         │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Analysis Layer                             │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                 Anomaly Detection Agent                     │ │
│  │            (src/agents/anomaly_detection_agent.py)          │ │
│  └────────────────────────────────────────────────────────────┘ │
│                               │                                  │
│                               ▼                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                     LLM Integration                         │ │
│  │           (Llama-3.1-Nemotron-70B-Instruct)                 │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Remediation Layer                           │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                    Remediation Agent                        │ │
│  │                  (run_remediation.py)                       │ │
│  └────────────────────────────────────────────────────────────┘ │
│                               │                                  │
│                               ▼                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                     K8s Actions                             │ │
│  │          (Pod Restart, Scaling, Resource Updates)           │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Orchestration Layer                          │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                   Multi-Agent System                        │ │
│  │              (k8s_multi_agent_system.py)                    │ │
│  └────────────────────────────────────────────────────────────┘ │
│                               │                                  │
│                               ▼                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                 Command Line Interface                      │ │
│  │             (Argument Parsing &amp; Configuration)              │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## Component Details

### Data Collection Layer

#### dataset-generator.py (Main Data Collector)

**Purpose**: Collects metrics from Kubernetes pods and nodes via Kubernetes API and Prometheus.

**Implementation Details**:

- Uses `kubernetes` Python client to interact with Kubernetes API
- Optionally queries Prometheus for additional metrics
- Periodically collects data at specified intervals
- Applies preprocessing to standardize data format
- Outputs data to CSV file for analysis

**Key Classes/Functions**:

- `KubernetesMetricsCollector`: Core class that manages connections to K8s API
- `PrometheusMetricsCollector`: Optional component for fetching Prometheus metrics
- `collect_pod_metrics()`: Main method for gathering pod-level data
- `collect_node_metrics()`: Gathers node-level performance statistics
- `write_metrics_to_csv()`: Outputs collected data to structured CSV

**Configuration Options**:

- `--output-file`: Output file path (default: pod_metrics.csv)
- `--prometheus-url`: Prometheus server URL (default: http://localhost:9090)
- `--namespace`: Kubernetes namespace to monitor (default: default)
- `--interval`: Collection interval in seconds (default: 30)

#### run_agent_with_custom_metrics.py

**Purpose**: Extends metrics collection with custom metrics that may not be available through standard APIs.

**Implementation Details**:

- Uses custom queries or API endpoints defined by the user
- Can integrate with specialized monitoring systems
- Outputs to the same CSV format for unified processing

### Processing Layer

#### Dataset Generator Agent (src/agents/dataset_generator_agent.py)

**Purpose**: Processes collected metrics, performs preprocessing, and detects anomalies.

**Implementation Details**:

- Continuously watches CSV file for new data entries
- Maintains historical data for trend analysis
- Processes raw metrics into features suitable for the ML model
- Communicates with the anomaly detection model
- Generates structured insights based on model output

**Key Classes/Functions**:

- `DatasetGeneratorAgent`: Main agent class
- `preprocess_metrics()`: Prepares raw metrics for analysis
- `read_new_data()`: Monitors CSV file for updates
- `update_pod_metrics()`: Refreshes internal state with new data
- `detect_anomalies()`: Interfaces with the anomaly prediction model
- `generate_insights()`: Creates actionable insights from detection results

**Configuration Options**:

- `--input-file`: Source CSV file path
- `--watch-interval`: Seconds between file checks
- `--alert-threshold`: Probability threshold for anomaly alerts
- `-agent/--agent-only`: Run only analysis components
- `-fetch/--fetch-only`: Run only data collection

#### Anomaly Prediction Model (models/anomaly_prediction.py)

**Purpose**: ML model that detects anomalies in pod behavior patterns.

**Implementation Details**:

- LSTM-based neural network for time series analysis
- Trained on historical pod behavior patterns
- Uses statistical methods to identify deviations
- Returns anomaly probability scores for each pod

**Key Components**:

- Pre-trained TensorFlow/Keras model for anomaly detection
- Feature normalization using sklearn scalers
- Anomaly threshold calibration based on historical data
- Explanation generation for detected anomalies

### Analysis Layer

#### Anomaly Detection Agent (src/agents/anomaly_detection_agent.py)

**Purpose**: Analyzes anomalies more deeply and generates explanations.

**Implementation Details**:

- Works with the Dataset Generator Agent's output
- Provides detailed analysis of why anomalies occurred
- Generates recommendations for remediation
- Can use LLM integration for enhanced analysis

**Key Classes/Functions**:

- `AnomalyDetectionAgent`: Core analysis class
- `analyze_anomaly()`: Examines specific anomaly patterns
- `generate_recommendations()`: Creates specific remediation suggestions
- `store_historical_patterns()`: Maintains context across runs

**Command**:

```bash
python anomaly_detection_agent.py
```

#### LLM Integration

**Purpose**: Uses language models to provide natural language explanations and insights.

**Implementation Details**:

- Integrates with Llama-3.1-Nemotron-70B-Instruct model
- Uses LangChain and LangGraph for workflow orchestration
- Generates context-aware explanations and recommendations
- Provides deeper insights than rule-based systems

**Key Components**:

- LangChain for prompt construction and context management
- LangGraph for workflow orchestration
- Templates for different types of analysis
- Integration with the Dataset Generator Agent

### Remediation Layer

#### Remediation Agent (run_remediation.py)

**Purpose**: Executes approved remediation actions on the Kubernetes cluster.

**Implementation Details**:

- Receives recommendations from analysis layer
- Presents options to users for approval
- Executes Kubernetes API calls to implement fixes
- Tracks remediation outcomes and effectiveness

**Key Classes/Functions**:

- `RemediationAgent`: Main remediation class
- `suggest_remediation()`: Proposes fixes based on insights
- `execute_action()`: Performs the approved remediation
- `validate_outcome()`: Checks if the remediation was successful
- `rollback()`: Reverts changes if remediation fails

**Command**:

```bash
python remediation_agent.py
```

**Remediation Actions**:

- Pod restart/recreation
- Deployment scaling
- Resource limit adjustments
- Node cordon/drain operations
- Custom actions based on specific anomalies

**Configuration Options**:

- `--metrics-file`: Path to metrics CSV file
- `--watch-interval`: Check interval in seconds
- `--confidence-threshold`: Threshold for action recommendations
- `--dry-run`: Show recommendations without applying
- `--auto-approve`: Automatically approve actions (use with caution)

### Orchestration Layer

#### Multi-Agent System (k8s_multi_agent_system.py)

**Purpose**: Coordinates all system components and manages communication between agents.

**Implementation Details**:

- Uses multiprocessing to run multiple agents concurrently
- Manages shared state and communication channels
- Handles system startup, shutdown, and error recovery
- Provides unified CLI for controlling all system components

**Key Classes/Functions**:

- `K8sMultiAgentSystem`: Main orchestration class
- `start()`: Initializes and runs all system components
- `stop()`: Gracefully shuts down all components
- `load_config()`: Manages system-wide configuration
- `signal_handler()`: Handles process termination signals

**Configuration Options**:

- `--config-file`: Path to YAML configuration file
- `--agent-mode`: Operation mode (full, monitoring, remediation)
- `--log-level`: Logging level (debug, info, warning, error)
- `--ui`: Enable web UI dashboard (experimental)
- `--llm-api-key`: API key for Llama-3.1-Nemotron-70B-Instruct integration

## Data Flow

1. **Collection**: Kubernetes metrics are collected from cluster via API and Prometheus
2. **Storage**: Raw metrics are stored in CSV format (pod_metrics.csv)
3. **Processing**: The Dataset Generator Agent reads and processes this data
4. **Analysis**:
   - The agent applies the anomaly prediction model
   - Anomalies are detected and scored
   - Insights are generated (stored in pod_insights.json)
5. **Enhanced Analysis**:
   - The Anomaly Detection Agent performs deeper analysis
   - Optional LLM integration provides natural language insights
6. **Remediation**:
   - The Remediation Agent proposes corrective actions
   - User approves or rejects remediation
   - Approved actions are executed on the Kubernetes cluster
7. **Orchestration**:
   - The Multi-Agent System coordinates all components
   - Ensures proper flow of data and control between layers

## Implementation Details

### Technology Stack

- **Programming Language**: Python 3.8+
- **Kubernetes Interaction**: Official Kubernetes Python client
- **Metrics Collection**: Kubernetes API, Prometheus API
- **Data Storage**: CSV files, JSON files
- **Machine Learning**: TensorFlow, scikit-learn, NumPy, Pandas
- **LLM Integration**: Llama-3.1-Nemotron-70B-Instruct, LangChain, LangGraph
- **Process Management**: Python multiprocessing library
- **Configuration**: YAML, environment variables, command-line arguments

## Directory Structure

```
k8s/
├── config/                  # Configuration files
│   ├── agent_config.yaml    # Agent behavior settings
│   ├── model_config.yaml    # ML model parameters
│   └── prometheus.yaml      # Prometheus connection settings
├── data/                    # Data storage directory
│   ├── pod_metrics.csv      # Raw metrics data
│   └── pod_insights.json    # Anomaly insights
├── logs/                    # Log files
│   ├── k8s_monitoring.log   # System logs
│   └── k8s_remediation.log  # Remediation logs
├── models/                  # Machine learning models
│   ├── anomaly_prediction.py
│   └── model_artifacts/     # Trained model files
├── src/                     # Source code
│   ├── agents/              # Agent implementations
│   │   ├── anomaly_detection_agent.py
│   │   ├── dataset_generator_agent.py
│   │   └── k8s_multi_agent_system.py
│   └── utils/               # Shared utilities
├── tests/                   # Test cases
├── dataset-generator.py     # Data collection script
├── run_agent_with_custom_metrics.py # Custom metrics collection
├── run_monitoring.py        # Orchestration script
├── run_remediation.py       # Remediation script
└── requirements.txt         # Dependencies
```

## Integration Points

### Kubernetes Integration

The system integrates with Kubernetes through:

1. **Kubernetes API**: Direct API calls for pod/node metrics and management
2. **kubectl**: Shell commands for specific operations
3. **Prometheus**: For metrics collection
4. **kube-state-metrics**: For additional cluster state information

### LLM Integration

LLM integration is optional and provides:

1. **Enhanced Analysis**: Better understanding of anomaly patterns
2. **Natural Language Explanations**: Human-readable insights
3. **Contextual Recommendations**: Better remediation suggestions

Integration is through:

- Llama-3.1-Nemotron-70B-Instruct API with proper authentication
- LangChain for prompt construction and parsing
- LangGraph for multi-step reasoning workflows

## Configuration System

The system uses a layered configuration approach:

1. **Default Values**: Hardcoded baseline settings
2. **Configuration Files**: YAML files in the `config/` directory
3. **Environment Variables**: Override file-based configuration
4. **Command-Line Arguments**: Highest priority, override all other settings

### Configuration Files

- **config/prometheus.yaml**: Prometheus connection settings
- **config/agent_config.yaml**: Agent behavior settings
- **config/model_config.yaml**: ML model parameters

### Environment Variables

- `LLAMA_NEMOTRON_API_KEY`: Authentication for Llama-3.1-Nemotron-70B-Instruct API
- `KUBECONFIG`: Path to Kubernetes configuration
- `PROMETHEUS_URL`: Prometheus server address
- `ANOMALY_THRESHOLD`: Detection sensitivity

## Extension Points

The system is designed for extensibility:

### Custom Metrics Collection

- Implement new collectors in the Data Collection Layer
- Use the existing CSV format for compatibility
- Add the new metrics source to the orchestration layer

### New ML Models

- Create new model implementations in the `models/` directory
- Follow the existing interface for prediction methods
- Register the model with the Dataset Generator Agent

### Custom Remediation Actions

- Add new action types to the Remediation Agent
- Implement execution logic in the agent
- Add user prompts and validation logic

### LLM Customization

- Modify prompt templates in the Analysis Layer
- Add new LangGraph workflows for specialized analysis
- Integrate with different LLM providers

## Deployment Considerations

### Prerequisites

- Kubernetes cluster (managed or local via Minikube)
- Prometheus monitoring stack
- Python 3.8+ environment
- Network access to the Kubernetes API server
- Appropriate RBAC permissions for remediation actions

### Resource Requirements

- **CPU**: Minimum 2 cores recommended
- **Memory**: 4GB+ recommended (more for LLM integration)
- **Storage**: 1GB for application, metrics storage depends on cluster size
- **Network**: Stable connection to Kubernetes API and Prometheus

### Security Considerations

- **RBAC**: Use minimal permissions required for operation
- **API Keys**: Secure storage of Llama-3.1-Nemotron-70B-Instruct API keys
- **Remediation**: Consider using `--dry-run` mode in production
- **Auto-approval**: Avoid using `--auto-approve` in critical environments

### High Availability

- The system can be deployed in a redundant configuration
- Data collection and analysis can run on separate nodes
- Use external persistent storage for metrics data
- Implement watchdog processes for component monitoring

## Getting Started

### Clone and Set Up Environment

```bash
git clone https://github.com/yourusername/kubernetes-monitoring.git
cd kubernetes-monitoring
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

### Start Kubernetes Locally

```bash
minikube start --cpus=4 --memory=8192 --driver=docker
```

### Setup Prometheus

```bash
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update
kubectl create namespace monitoring
helm install prometheus prometheus-community/kube-prometheus-stack --namespace monitoring
kubectl port-forward -n monitoring svc/prometheus-kube-prometheus-prometheus 9090:9090
```

### Run the System

**Run all components:**

```bash
python run_monitoring.py
```

**Run the multi-agent system:**

```bash
cd src/agents
python k8s_multi_agent_system.py
```

**Run individual components:**

```bash
python dataset-generator.py
python src/agents/dataset_generator_agent.py
python anomaly_detection_agent.py
python remediation_agent.py
```

### Environment Variables

```bash
export LLAMA_NEMOTRON_API_KEY=your-llama-nemotron-api-key
export KUBECONFIG=/path/to/config
export ANOMALY_THRESHOLD=0.8
```

### Outputs

- **pod_metrics.csv**: Raw metrics data
- **pod_insights.json**: Anomaly insights
- **k8s_monitoring.log**: System logs

### Troubleshooting

- **Connection Issues**: Check `kubectl config view` and `kubectl cluster-info`
- **Missing Metrics**: Verify Prometheus setup and port-forwarding
- **Model Errors**: Ensure model files exist in expected locations
- **Remediation Failures**: Check k8s_remediation.log for error details

<div>⁂</div>

[^1]: https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/60031456/9b52c43b-848b-4c20-9ad5-cbbcaccb3a18/ARCHITECTURE.md
