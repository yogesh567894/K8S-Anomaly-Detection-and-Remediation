#!/usr/bin/env python3
"""
Kubernetes Anomaly Visualization Dashboard

A Streamlit-based visualization dashboard for the Kubernetes multi-agent system.
Features:
- Cluster topology visualization with anomaly highlighting
- Time-series metrics visualization
- AI-powered insights panel using LLM
- Interactive remediation workflow
"""

import os
import sys
import json
import time
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import networkx as nx
from streamlit_plotly_events import plotly_events
import subprocess
import threading
import queue

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from the multi-agent system
try:
    from agents.k8s_multi_agent_system import collect_pod_metrics, predict_pod_anomaly
    from agents.nvidia_llm import NvidiaLLM
except ImportError:
    st.error("Failed to import required modules. Make sure you're running from the project root.")
    st.stop()

# Initialize LLM for AI insights
try:
    nvidia_api_key = os.environ.get("NVIDIA_API_KEY", "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")
    llm = NvidiaLLM(api_key=nvidia_api_key)
except Exception as e:
    st.warning(f"Could not initialize LLM: {str(e)}. Some AI features will be limited.")
    llm = None

# Metrics collection and processing
def get_metrics(test_mode=False):
    """Get current Kubernetes metrics"""
    if test_mode:
        # Load sample data from CSV
        try:
            metrics_df = pd.read_csv("src/agents/pod_metrics.csv")
            pod_metrics = {}
            for _, row in metrics_df.iterrows():
                pod_name = row['Pod Name']
                pod_metrics[pod_name] = row.to_dict()
            return pod_metrics
        except Exception as e:
            st.error(f"Error loading test metrics: {str(e)}")
            return {}
    else:
        # Get real metrics
        try:
            return collect_pod_metrics()
        except Exception as e:
            st.error(f"Error collecting metrics: {str(e)}")
            return {}

def detect_anomalies(pod_metrics):
    """Detect anomalies in pod metrics"""
    anomalies = {}
    for pod_name, metrics in pod_metrics.items():
        try:
            is_anomaly, prediction = predict_pod_anomaly(metrics)
            if is_anomaly:
                anomalies[pod_name] = {
                    'metrics': metrics,
                    'prediction': prediction
                }
        except Exception as e:
            st.warning(f"Error detecting anomalies for pod {pod_name}: {str(e)}")
    return anomalies

def get_cluster_graph(pod_metrics):
    """Create a graph representation of the cluster"""
    G = nx.Graph()
    
    # Add nodes
    nodes = {}
    
    # First add the nodes (Kubernetes nodes)
    k8s_nodes = set()
    for pod_name, metrics in pod_metrics.items():
        node_name = metrics.get('node', 'unknown')
        k8s_nodes.add(node_name)
    
    # Add Kubernetes nodes
    for node in k8s_nodes:
        G.add_node(node, type='node', name=node)
        nodes[node] = {'type': 'node', 'name': node}
    
    # Then add pods and connect to nodes
    for pod_name, metrics in pod_metrics.items():
        node_name = metrics.get('node', 'unknown')
        G.add_node(pod_name, type='pod', name=pod_name)
        nodes[pod_name] = {'type': 'pod', 'name': pod_name}
        G.add_edge(pod_name, node_name)
    
    # Use Fruchterman-Reingold layout
    pos = nx.spring_layout(G)
    
    # Convert to visualization format
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    node_x = []
    node_y = []
    node_text = []
    node_color = []
    node_size = []
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        
        if nodes[node]['type'] == 'node':
            node_text.append(f"Node: {node}")
            node_color.append('blue')
            node_size.append(20)
        else:
            node_text.append(f"Pod: {node}")
            node_color.append('green')
            node_size.append(15)
    
    return {
        'edge_x': edge_x,
        'edge_y': edge_y,
        'node_x': node_x,
        'node_y': node_y,
        'node_text': node_text,
        'node_color': node_color,
        'node_size': node_size,
        'nodes': list(G.nodes()),
    }

def highlight_anomalies(graph, anomalies):
    """Highlight anomalous pods in the cluster graph"""
    for i, node in enumerate(graph['nodes']):
        if node in anomalies:
            graph['node_color'][i] = 'red'
            graph['node_size'][i] = 25
    return graph

def execute_remediation(pod_name, action):
    """Execute remediation action on a Kubernetes pod
    
    Args:
        pod_name: Name of the pod to remediate
        action: Type of remediation action to perform
        
    Returns:
        tuple: (success_bool, detail_message)
    """
    # Log the remediation attempt
    print(f"Attempting remediation: {action} on pod {pod_name}")
    
    # Validate inputs
    if not pod_name or not action:
        return False, "Invalid pod name or action"
    
    # Map actions to more descriptive labels for UI feedback
    action_descriptions = {
        "restart_pod": "Restarting pod",
        "restart_deployment": "Restarting the deployment",
        "increase_memory": "Increasing memory allocation",
        "increase_cpu": "Scaling up CPU resources",
        "scale_deployment": "Scaling deployment replicas"
    }
    
    action_desc = action_descriptions.get(action, action)
    
    # Build the remediation command
    command = f"python src/agents/remediation_dashboard_agent.py --pod {pod_name} --action {action}"
    
    def run_command(cmd, result_queue):
        try:
            # Run the command with a timeout
            result = subprocess.run(
                cmd, 
                shell=True, 
                capture_output=True, 
                text=True
            )
            result_queue.put((result.returncode, result.stdout, result.stderr))
        except subprocess.SubprocessError as e:
            result_queue.put((1, "", f"Subprocess error: {str(e)}"))
        except Exception as e:
            result_queue.put((1, "", f"Unexpected error: {str(e)}"))
    
    # Create a queue for thread communication
    result_queue = queue.Queue()
    
    # Run the command in a separate thread
    thread = threading.Thread(target=run_command, args=(command, result_queue))
    thread.start()
    
    # Wait for command to complete with timeout
    timeout = 30  # seconds
    thread.join(timeout=timeout)
    
    # Check if thread is still running (timed out)
    if thread.is_alive():
        return False, f"Remediation timed out after {timeout} seconds"
    
    # Get results from queue
    try:
        code, stdout, stderr = result_queue.get(block=False)
        
        if code == 0:
            # Format successful output message
            message = stdout.strip() if stdout.strip() else f"Successfully executed: {action_desc}"
            return True, message
        else:
            # Format error message
            if stderr.strip():
                error_msg = f"Error: {stderr.strip()}"
            else:
                error_msg = f"Failed to execute {action_desc} (Exit code: {code})"
            return False, error_msg
            
    except queue.Empty:
        return False, "No result returned from remediation command"

def get_ai_analysis(pod_metrics, anomalies):
    """Get AI-powered analysis of pod metrics and anomalies"""
    if not llm:
        return "AI analysis not available (LLM not initialized)"
    
    if not anomalies:
        return "No anomalies detected. The cluster appears to be healthy."
    
    # Generate a prompt for the LLM but with more concise instructions
    prompt = "Analyze these Kubernetes metrics and detected anomalies:\n\n"
    
    for pod_name, anomaly_data in anomalies.items():
        metrics = anomaly_data['metrics']
        prediction = anomaly_data['prediction']
        
        prompt += f"Pod: {pod_name}\n"
        prompt += f"Anomaly Type: {prediction.get('anomaly_type', 'unknown')}\n"
        prompt += f"Confidence: {prediction.get('anomaly_probability', 0):.2f}\n"
        prompt += f"CPU: {metrics.get('CPU Usage (%)', 'N/A')}%, Memory: {metrics.get('Memory Usage (%)', 'N/A')}%, Restarts: {metrics.get('Pod Restarts', 'N/A')}\n"
        
        if 'Event Reason' in metrics:
            prompt += f"Event: {metrics.get('Event Reason', 'N/A')} - {metrics.get('Event Message', 'N/A')}\n"
        
        prompt += "\n"
    
    prompt += "Provide a VERY BRIEF analysis (max 3-4 sentences) that summarizes:\n"
    prompt += "1. Root cause of anomalies\n"
    prompt += "2. Severity level\n"
    prompt += "3. Recommended actions\n"
    
    try:
        analysis = llm.generate(prompt, temperature=0.3, max_tokens=200)  # Limit token length for brevity
        return analysis
    except Exception as e:
        return f"Error generating AI analysis: {str(e)}"

# Streamlit UI
def build_ui():
    st.set_page_config(
        page_title="Kubernetes Anomaly Visualization Dashboard",
        page_icon="🔍",
        layout="wide"
    )
    
    st.title("Kubernetes Anomaly Visualization Dashboard")
    
    # Sidebar with options
    with st.sidebar:
        st.header("Options")
        test_mode = st.checkbox("Test Mode (Use sample data)", value=True)
        refresh_interval = st.slider("Refresh Interval (seconds)", min_value=10, max_value=300, value=60)
        auto_refresh = st.checkbox("Auto Refresh", value=False)
        
        st.header("Actions")
        if st.button("Refresh Data"):
            st.experimental_rerun()
        
        st.header("Settings")
        anomaly_threshold = st.slider("Anomaly Confidence Threshold", min_value=0.5, max_value=0.99, value=0.7)
        st.write("---")
        st.write("**Visualization Settings**")
        visualization_type = st.selectbox(
            "Chart Type", 
            ["Area Chart", "Line Chart", "Bar Chart", "Scatter Plot"],
            index=0
        )
        animation_speed = st.slider("Animation Speed", min_value=300, max_value=2000, value=800)
    
    # Main layout - using single column instead of two columns
    
    # Get data
    pod_metrics = get_metrics(test_mode=test_mode)
    
    if not pod_metrics:
        st.error("No metrics data available. Check your Kubernetes connection or sample data.")
        return
    
    # Detect anomalies
    anomalies = detect_anomalies(pod_metrics)
    
    # Create interactive metrics visualization
    st.subheader("Interactive Pod Metrics")
    
    # Convert pod metrics to time series data and add synthetic time data for testing
    # In a real implementation, we'd have real time series data from Prometheus or similar
    current_time = datetime.now()
    time_points = [current_time - timedelta(minutes=i*5) for i in range(12)]
    time_points.reverse()  # So they go from oldest to newest
    
    # Create synthetic historical data for testing
    metrics_time_series = []
    for pod_name, metrics in pod_metrics.items():
        base_cpu = float(metrics.get('CPU Usage (%)', 10))
        base_memory = float(metrics.get('Memory Usage (%)', 20))
        
        # Check if this pod has anomalies
        is_anomalous = pod_name in anomalies
        
        for i, timestamp in enumerate(time_points):
            # Add some randomness but make anomalous pods trend upward
            if is_anomalous and i > 6:  # Anomalies start after the 6th time point
                cpu_factor = 1.2 + (i-6)*0.3  # Increasing trend
                memory_factor = 1.15 + (i-6)*0.25  # Increasing trend
            else:
                cpu_factor = 0.9 + np.random.random() * 0.2
                memory_factor = 0.85 + np.random.random() * 0.3
            
            metrics_time_series.append({
                'Pod': pod_name,
                'Timestamp': timestamp,
                'CPU': min(base_cpu * cpu_factor, 100),  # Cap at 100%
                'Memory': min(base_memory * memory_factor, 100),  # Cap at 100%
                'Status': 'Anomalous' if is_anomalous and i > 6 else 'Normal',
                'Restarts': metrics.get('Pod Restarts', 0) if i > 9 else max(0, int(metrics.get('Pod Restarts', 0)) - 1)
            })
    
    metrics_df = pd.DataFrame(metrics_time_series)
    
    # Create tabs for different metric views
    tab1, tab2, tab3, tab4 = st.tabs(["CPU Usage", "Memory Usage", "Pod Status", "Combined View"])
    
    with tab1:
        # Interactive CPU usage chart with anomaly highlighting
        if visualization_type == "Area Chart":
            fig = px.area(
                metrics_df, 
                x="Timestamp", 
                y="CPU", 
                color="Pod",
                line_group="Pod",
                color_discrete_sequence=px.colors.qualitative.Bold,
                hover_data=["Status", "Restarts"],
                title="CPU Usage Over Time (%)"
            )
        elif visualization_type == "Line Chart":
            fig = px.line(
                metrics_df, 
                x="Timestamp", 
                y="CPU", 
                color="Pod",
                line_dash="Status",  # Dashed lines for anomalous pods
                symbols="Status",
                hover_data=["Status", "Restarts"],
                title="CPU Usage Over Time (%)"
            )
        elif visualization_type == "Bar Chart":
            fig = px.bar(
                metrics_df, 
                x="Timestamp", 
                y="CPU",
                color="Pod",
                hover_data=["Status", "Restarts"],
                title="CPU Usage Over Time (%)"
            )
        else:  # Scatter plot
            fig = px.scatter(
                metrics_df, 
                x="Timestamp", 
                y="CPU",
                color="Pod",
                size="Restarts",
                symbol="Status",
                hover_data=["Status", "Restarts"],
                title="CPU Usage Over Time (%)"
            )
        
        # Add threshold line for CPU usage
        fig.add_shape(
            type="line",
            x0=metrics_df["Timestamp"].min(),
            y0=80,
            x1=metrics_df["Timestamp"].max(),
            y1=80,
            line=dict(color="Red", width=2, dash="dash"),
        )
        
        # Highlight anomalous regions
        for pod_name in anomalies:
            pod_data = metrics_df[metrics_df['Pod'] == pod_name]
            anomalous_data = pod_data[pod_data['Status'] == 'Anomalous']
            if not anomalous_data.empty:
                first_anomaly = anomalous_data['Timestamp'].min()
                fig.add_vrect(
                    x0=first_anomaly,
                    x1=metrics_df['Timestamp'].max(),
                    fillcolor="red",
                    opacity=0.1,
                    layer="below",
                    line_width=0,
                )
        
        # Enable animations for a more interactive feel
        fig.update_layout(
            transition_duration=animation_speed,
            hovermode="closest",
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        # Add annotations for anomalies
        for pod_name in anomalies:
            pod_data = metrics_df[metrics_df['Pod'] == pod_name]
            last_point = pod_data.iloc[-1]
            
            fig.add_annotation(
                x=last_point['Timestamp'],
                y=last_point['CPU'],
                text=f"Anomaly: {pod_name}",
                showarrow=True,
                arrowhead=1,
                ax=-40,
                ay=-40
            )
        
        st.plotly_chart(fig, use_container_width=True)
        
        with st.expander("CPU Usage Details"):
            st.dataframe(
                metrics_df.pivot(index='Timestamp', columns='Pod', values='CPU').reset_index(),
                use_container_width=True
            )
    
    with tab2:
        # Interactive Memory usage chart
        if visualization_type == "Area Chart":
            fig = px.area(
                metrics_df, 
                x="Timestamp", 
                y="Memory", 
                color="Pod",
                line_group="Pod",
                color_discrete_sequence=px.colors.qualitative.Vivid,
                hover_data=["Status", "Restarts"],
                title="Memory Usage Over Time (%)"
            )
        elif visualization_type == "Line Chart":
            fig = px.line(
                metrics_df, 
                x="Timestamp", 
                y="Memory", 
                color="Pod",
                line_dash="Status",
                symbols="Status",
                hover_data=["Status", "Restarts"],
                title="Memory Usage Over Time (%)"
            )
        elif visualization_type == "Bar Chart":
            fig = px.bar(
                metrics_df, 
                x="Timestamp", 
                y="Memory",
                color="Pod",
                hover_data=["Status", "Restarts"],
                title="Memory Usage Over Time (%)"
            )
        else:  # Scatter plot
            fig = px.scatter(
                metrics_df, 
                x="Timestamp", 
                y="Memory",
                color="Pod",
                size="Restarts",
                symbol="Status",
                hover_data=["Status", "Restarts"],
                title="Memory Usage Over Time (%)"
            )
        
        # Add threshold line
        fig.add_shape(
            type="line",
            x0=metrics_df["Timestamp"].min(),
            y0=85,
            x1=metrics_df["Timestamp"].max(),
            y1=85,
            line=dict(color="Red", width=2, dash="dash"),
        )
        
        # Highlight anomalous regions
        for pod_name in anomalies:
            pod_data = metrics_df[metrics_df['Pod'] == pod_name]
            anomalous_data = pod_data[pod_data['Status'] == 'Anomalous']
            if not anomalous_data.empty:
                first_anomaly = anomalous_data['Timestamp'].min()
                fig.add_vrect(
                    x0=first_anomaly,
                    x1=metrics_df['Timestamp'].max(),
                    fillcolor="red",
                    opacity=0.1,
                    layer="below",
                    line_width=0,
                )
        
        # Enable animations
        fig.update_layout(
            transition_duration=animation_speed,
            hovermode="closest",
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        with st.expander("Memory Usage Details"):
            st.dataframe(
                metrics_df.pivot(index='Timestamp', columns='Pod', values='Memory').reset_index(),
                use_container_width=True
            )
    
    with tab3:
        # Pod status timeline
        restarts_df = metrics_df.pivot(index='Timestamp', columns='Pod', values='Restarts').reset_index()
        status_df = metrics_df.pivot(index='Timestamp', columns='Pod', values='Status').reset_index()
        
        st.write("Pod Status Timeline")
        
        fig = go.Figure()
        
        for pod_name in metrics_df['Pod'].unique():
            pod_data = metrics_df[metrics_df['Pod'] == pod_name]
            
            # Create a line for normal status
            normal_data = pod_data[pod_data['Status'] == 'Normal']
            if not normal_data.empty:
                fig.add_trace(go.Scatter(
                    x=normal_data['Timestamp'],
                    y=[pod_name] * len(normal_data),
                    mode='lines',
                    name=f"{pod_name} (Normal)",
                    line=dict(color='green', width=10),
                    hoverinfo='text',
                    hovertext=[f"Pod: {pod_name}<br>Status: Normal<br>Time: {ts}" for ts in normal_data['Timestamp']]
                ))
            
            # Create a line for anomalous status
            anomalous_data = pod_data[pod_data['Status'] == 'Anomalous']
            if not anomalous_data.empty:
                fig.add_trace(go.Scatter(
                    x=anomalous_data['Timestamp'],
                    y=[pod_name] * len(anomalous_data),
                    mode='lines',
                    name=f"{pod_name} (Anomalous)",
                    line=dict(color='red', width=10),
                    hoverinfo='text',
                    hovertext=[f"Pod: {pod_name}<br>Status: Anomalous<br>Time: {ts}" for ts in anomalous_data['Timestamp']]
                ))
                
            # Add restart markers
            restart_data = pod_data[pod_data['Restarts'] > 0]
            if not restart_data.empty:
                fig.add_trace(go.Scatter(
                    x=restart_data['Timestamp'],
                    y=[pod_name] * len(restart_data),
                    mode='markers',
                    marker=dict(symbol='x', size=12, color='black'),
                    name=f"{pod_name} (Restart)",
                    hoverinfo='text',
                    hovertext=[f"Pod: {pod_name}<br>Restarted at: {ts}<br>Restart count: {r}" 
                              for ts, r in zip(restart_data['Timestamp'], restart_data['Restarts'])]
                ))
        
        fig.update_layout(
            title="Pod Status Timeline",
            xaxis_title="Time",
            yaxis_title="Pod",
            hovermode="closest",
            height=400,
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        with st.expander("Status Details"):
            st.write("Pod Restart Count Over Time")
            st.dataframe(restarts_df, use_container_width=True)
            
            st.write("Pod Status Over Time")
            st.dataframe(status_df, use_container_width=True)
    
    with tab4:
        # Combined view with multiple metrics on same chart
        st.write("Combined Resource Usage")
        
        # Get the last data point for each pod
        latest_metrics = metrics_df.sort_values('Timestamp').groupby('Pod').last().reset_index()
        
        # Create a bubble chart with CPU, Memory, and Size based on restart count
        fig = px.scatter(
            latest_metrics,
            x="CPU",
            y="Memory",
            size="Restarts",
            color="Status",
            hover_name="Pod",
            text="Pod",
            size_max=40,
            color_discrete_map={"Normal": "green", "Anomalous": "red"},
        )
        
        fig.update_layout(
            title="Pod Resource Usage (Latest)",
            xaxis_title="CPU Usage (%)",
            yaxis_title="Memory Usage (%)",
            height=500,
        )
        
        # Add quadrant lines
        fig.add_shape(type="line", x0=0, y0=80, x1=100, y1=80, line=dict(color="Red", width=1, dash="dash"))
        fig.add_shape(type="line", x0=80, y0=0, x1=80, y1=100, line=dict(color="Red", width=1, dash="dash"))
        
        # Add quadrant labels
        fig.add_annotation(x=40, y=40, text="Healthy", showarrow=False, font=dict(size=14, color="green"))
        fig.add_annotation(x=90, y=40, text="High CPU", showarrow=False, font=dict(size=14, color="orange"))
        fig.add_annotation(x=40, y=90, text="High Memory", showarrow=False, font=dict(size=14, color="orange"))
        fig.add_annotation(x=90, y=90, text="Critical", showarrow=False, font=dict(size=14, color="red"))
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Show detected anomalies in a more compact format
    st.subheader("Detected Anomalies")
    
    if anomalies:
        st.error(f"{len(anomalies)} anomalies detected!")
        
        # Create columns for compact display
        columns = st.columns(min(3, len(anomalies)))
        
        for i, (pod_name, anomaly_data) in enumerate(anomalies.items()):
            column_index = i % len(columns)
            
            with columns[column_index]:
                st.warning(f"🚨 {pod_name}")
                prediction = anomaly_data['prediction']
                metrics = anomaly_data['metrics']
                
                st.write(f"**Type:** {prediction.get('anomaly_type', 'unknown')}")
                st.write(f"**Confidence:** {prediction.get('anomaly_probability', 0):.2f}")
                st.write(f"**CPU:** {metrics.get('CPU Usage (%)', 'N/A')}% | **Mem:** {metrics.get('Memory Usage (%)', 'N/A')}%")
                
                # Compact remediation dropdown and button
                action = st.selectbox("Action", [
                    "restart_pod",
                    "restart_deployment",
                    "increase_memory",
                    "increase_cpu",
                    "scale_deployment"
                ], key=f"action_{pod_name}")
                
                if st.button("Execute", key=f"remediate_{pod_name}"):
                    with st.spinner(f"Executing {action}..."):
                        success, message = execute_remediation(pod_name, action)
                        if success:
                            st.success("✓")
                        else:
                            st.error("✗")
    else:
        st.success("No anomalies detected. The cluster appears to be healthy!")
    
    # AI Insights panel - now more concise
    st.subheader("AI-Powered Insights")
    
    with st.spinner("Generating insights..."):
        analysis = get_ai_analysis(pod_metrics, anomalies)
        st.info(analysis)
    
    # Footer with metadata
    with st.expander("Monitoring Metadata"):
        col1, col2, col3 = st.columns(3)
        col1.info(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        col2.info(f"Pods monitored: {len(pod_metrics)}")
        col3.info(f"Anomalies detected: {len(anomalies)}")
    
    if auto_refresh:
        time.sleep(refresh_interval)
        st.experimental_rerun()

if __name__ == "__main__":
    build_ui() 