#!/usr/bin/env python3
"""
Kubernetes Remediation Runner Script

This script monitors for anomalies and calls the remediation agent
when needed to take corrective action with user approval.
"""

import os
import sys
import time
import argparse
import signal
import logging
from datetime import datetime
import pandas as pd

# Import our custom modules
from anomaly_prediction import predict_anomalies
from remediation_agent import remediate_pod

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('k8s_remediation.log')
    ]
)
logger = logging.getLogger("k8s-remediation-runner")

# Global variables for process management
stop_event = False

def signal_handler(sig, frame):
    """Handle termination signals to gracefully shut down"""
    global stop_event
    logger.info("Received termination signal, shutting down...")
    stop_event = True
    sys.exit(0)

def get_pod_info_from_metrics(pod_name, namespace, metrics_data):
    """Extract pod info from metrics data"""
    pod_info = {
        'name': pod_name,
        'namespace': namespace,
        'status': metrics_data.get('Pod Status', 'Unknown'),
        'restart_count': metrics_data.get('Pod Restarts', 0),
    }
    
    # Add resource information if available
    if 'CPU Request' in metrics_data:
        pod_info['cpu_request'] = metrics_data['CPU Request']
    if 'Memory Request' in metrics_data:
        pod_info['memory_request'] = metrics_data['Memory Request']
    if 'CPU Limit' in metrics_data:
        pod_info['cpu_limit'] = metrics_data['CPU Limit']
    if 'Memory Limit' in metrics_data:
        pod_info['memory_limit'] = metrics_data['Memory Limit']
    if 'Node' in metrics_data:
        pod_info['node'] = metrics_data['Node']
    if 'Pod IP' in metrics_data:
        pod_info['ip'] = metrics_data['Pod IP']
        
    # Try to extract owner reference (deployment name)
    if pod_name and '-' in pod_name:
        # Heuristic: Most deployment pods have format name-randomstring
        possible_owner = pod_name.rsplit('-', 1)[0]
        pod_info['owner_reference'] = possible_owner
    
    return pod_info

def run_remediation_loop(metrics_file, watch_interval, confidence_threshold, auto_approve=False):
    """Run the main remediation loop monitoring metrics file for anomalies"""
    global stop_event
    
    logger.info(f"Starting remediation loop with metrics file: {metrics_file}")
    logger.info(f"Watch interval: {watch_interval}s, confidence threshold: {confidence_threshold}")
    logger.info(f"Auto-approve: {auto_approve}")
    
    # Track which pods we've already processed to avoid duplicate alerts
    processed_pods = {}
    
    # Track last modification time of metrics file
    last_mtime = 0
    
    try:
        while not stop_event:
            try:
                # Check if metrics file exists and has been updated
                if not os.path.exists(metrics_file):
                    logger.warning(f"Metrics file {metrics_file} not found, waiting...")
                    time.sleep(watch_interval)
                    continue
                
                current_mtime = os.path.getmtime(metrics_file)
                if current_mtime <= last_mtime:
                    # File hasn't been modified, wait for next check
                    time.sleep(watch_interval)
                    continue
                
                last_mtime = current_mtime
                
                # Read the metrics data
                metrics_df = pd.read_csv(metrics_file)
                if metrics_df.empty:
                    logger.warning("Metrics file is empty, waiting for data...")
                    time.sleep(watch_interval)
                    continue
                
                logger.info(f"Processing {len(metrics_df)} metric entries")
                
                # Group by pod name and namespace
                if 'Namespace' in metrics_df.columns:
                    pod_groups = metrics_df.groupby(['Pod Name', 'Namespace'])
                else:
                    # If namespace column is missing, assume default namespace
                    metrics_df['Namespace'] = 'default'
                    pod_groups = metrics_df.groupby(['Pod Name', 'Namespace'])
                
                # Process each pod's metrics
                for (pod_name, namespace), pod_metrics in pod_groups:
                    # Skip empty pod names
                    if not pod_name:
                        continue
                        
                    # Use the latest metrics for this pod
                    latest_metrics = pod_metrics.iloc[-1].to_dict()
                    
                    # Get pod info
                    pod_info = get_pod_info_from_metrics(pod_name, namespace, latest_metrics)
                    
                    # Convert to dataframe for prediction
                    metrics_for_prediction = pd.DataFrame([latest_metrics])
                    
                    # Predict anomalies
                    prediction_df = predict_anomalies(metrics_for_prediction)
                    prediction = prediction_df.iloc[0].to_dict()
                    
                    # Check if it's an anomaly with sufficient confidence
                    is_anomaly = prediction.get('predicted_anomaly', 0) == 1
                    probability = prediction.get('anomaly_probability', 0.0)
                    
                    if is_anomaly and probability >= confidence_threshold:
                        # Generate a unique ID for this anomaly
                        anomaly_time = datetime.now().strftime("%Y%m%d-%H%M%S")
                        anomaly_id = f"{namespace}-{pod_name}-{anomaly_time}"
                        
                        # Check cooldown - don't process the same pod too frequently
                        last_processed = processed_pods.get(f"{namespace}/{pod_name}", 0)
                        current_time = time.time()
                        cooldown_period = 300  # 5 minutes
                        
                        if current_time - last_processed < cooldown_period:
                            logger.info(f"Pod {namespace}/{pod_name} in cooldown period, skipping")
                            continue
                        
                        # Add event information to prediction
                        prediction['event_age_minutes'] = latest_metrics.get('Event Age (minutes)', 0)
                        prediction['event_reason'] = latest_metrics.get('Event Reason', '')
                        prediction['event_count'] = latest_metrics.get('Event Count', 0)
                        
                        logger.info(f"Anomaly detected for pod {namespace}/{pod_name}: "
                                   f"{prediction.get('anomaly_type', 'unknown')} "
                                   f"(confidence: {probability:.4f})")
                        
                        # Call the remediation agent
                        logger.info(f"Calling remediation agent for pod {namespace}/{pod_name}")
                        
                        # If auto-approve is enabled, provide "yes" as user input
                        user_input = "yes" if auto_approve else None
                        
                        messages = remediate_pod(prediction, pod_info, user_input)
                        
                        # Print messages to console
                        print("\n" + "="*80)
                        print(f"ANOMALY DETECTED: {namespace}/{pod_name}")
                        print("="*80)
                        
                        for msg in messages:
                            from langchain_core.messages import AIMessage
                            prefix = "AI: " if isinstance(msg, AIMessage) else "Human: "
                            content = msg.content if hasattr(msg, 'content') else str(msg)
                            print(f"{prefix}{content}")
                        
                        # If not auto-approve, ask for user input
                        if not auto_approve:
                            user_response = input("\nDo you approve this remediation plan? (yes/no): ")
                            
                            if user_response.lower() in ["yes", "y", "approve", "approved"]:
                                print("\nExecuting remediation plan...")
                                messages = remediate_pod(prediction, pod_info, user_response)
                                
                                # Print new messages (only the ones added after approval)
                                for msg in messages[len(messages)-2:]:
                                    prefix = "AI: " if isinstance(msg, AIMessage) else "Human: "
                                    content = msg.content if hasattr(msg, 'content') else str(msg)
                                    print(f"{prefix}{content}")
                            else:
                                print("\nRemediation plan rejected. No action taken.")
                        
                        # Update processed pods with timestamp
                        processed_pods[f"{namespace}/{pod_name}"] = current_time
                
            except Exception as e:
                logger.error(f"Error in remediation loop: {e}")
                import traceback
                traceback.print_exc()
            
            # Wait for next check
            time.sleep(watch_interval)
            
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down")
    finally:
        logger.info("Remediation loop stopped")

def main():
    """Parse arguments and start monitoring"""
    parser = argparse.ArgumentParser(description='Run Kubernetes remediation with anomaly detection')
    
    # Remediation options
    parser.add_argument('--metrics-file', type=str, default='pod_metrics.csv',
                        help='Input file with metrics (default: pod_metrics.csv)')
    parser.add_argument('--watch-interval', type=int, default=10,
                        help='Interval in seconds between checks (default: 10)')
    parser.add_argument('--confidence-threshold', type=float, default=0.7,
                        help='Probability threshold for anomaly alerts (default: 0.7)')
    parser.add_argument('--auto-approve', action='store_true',
                        help='Automatically approve remediation actions (USE WITH CAUTION)')
    
    args = parser.parse_args()
    
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Start the remediation loop
        run_remediation_loop(
            args.metrics_file,
            args.watch_interval,
            args.confidence_threshold,
            args.auto_approve
        )
    except Exception as e:
        logger.error(f"Error in main process: {e}")
        import traceback
        traceback.print_exc()
    
    logger.info("Remediation agent stopped")

if __name__ == "__main__":
    main() 