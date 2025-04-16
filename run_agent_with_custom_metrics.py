import pandas as pd
import argparse
import json
from simple_ai_agent import SimpleAIAgent

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Run AI agent with custom metrics')
    parser.add_argument('--metrics-file', type=str, help='Path to JSON file containing metrics')
    parser.add_argument('--cpu-usage', type=float, help='CPU Usage (%)')
    parser.add_argument('--memory-usage', type=float, help='Memory Usage (%)')
    parser.add_argument('--pod-restarts', type=int, help='Pod Restarts')
    parser.add_argument('--memory-usage-mb', type=float, help='Memory Usage (MB)')
    parser.add_argument('--network-receive-bytes', type=float, help='Network Receive Bytes')
    parser.add_argument('--network-transmit-bytes', type=float, help='Network Transmit Bytes')
    parser.add_argument('--fs-reads-total-mb', type=float, help='FS Reads Total (MB)')
    parser.add_argument('--fs-writes-total-mb', type=float, help='FS Writes Total (MB)')
    parser.add_argument('--network-receive-packets-dropped', type=float, help='Network Receive Packets Dropped (p/s)')
    parser.add_argument('--network-transmit-packets-dropped', type=float, help='Network Transmit Packets Dropped (p/s)')
    parser.add_argument('--ready-containers', type=int, help='Ready Containers')
    return parser.parse_args()

def load_metrics_from_file(file_path):
    """Load metrics from a JSON file"""
    with open(file_path, 'r') as f:
        metrics = json.load(f)
    return metrics

def create_metrics_from_args(args):
    """Create metrics dictionary from command line arguments"""
    metrics = {}
    
    if args.cpu_usage is not None:
        metrics['CPU Usage (%)'] = args.cpu_usage
    if args.memory_usage is not None:
        metrics['Memory Usage (%)'] = args.memory_usage
    if args.pod_restarts is not None:
        metrics['Pod Restarts'] = args.pod_restarts
    if args.memory_usage_mb is not None:
        metrics['Memory Usage (MB)'] = args.memory_usage_mb
    if args.network_receive_bytes is not None:
        metrics['Network Receive Bytes'] = args.network_receive_bytes
    if args.network_transmit_bytes is not None:
        metrics['Network Transmit Bytes'] = args.network_transmit_bytes
    if args.fs_reads_total_mb is not None:
        metrics['FS Reads Total (MB)'] = args.fs_reads_total_mb
    if args.fs_writes_total_mb is not None:
        metrics['FS Writes Total (MB)'] = args.fs_writes_total_mb
    if args.network_receive_packets_dropped is not None:
        metrics['Network Receive Packets Dropped (p/s)'] = args.network_receive_packets_dropped
    if args.network_transmit_packets_dropped is not None:
        metrics['Network Transmit Packets Dropped (p/s)'] = args.network_transmit_packets_dropped
    if args.ready_containers is not None:
        metrics['Ready Containers'] = args.ready_containers
    
    return metrics

def main():
    """Main function"""
    args = parse_arguments()
    
    # Get metrics from file or command line arguments
    if args.metrics_file:
        metrics = load_metrics_from_file(args.metrics_file)
    else:
        metrics = create_metrics_from_args(args)
    
    # Create and run the agent
    agent = SimpleAIAgent()
    
    # Override the collect_metrics method to use our custom metrics
    def custom_collect_metrics(self):
        """Collect metrics from the provided data"""
        self.current_metrics = pd.DataFrame([metrics])
    
    # Replace the collect_metrics method
    agent.collect_metrics = custom_collect_metrics.__get__(agent)
    
    # Run the agent
    result = agent.run()
    
    # Print the results
    print("\nAI Agent Prediction Results:")
    print(f"Anomaly Detected: {result['anomaly_detected']}")
    print(f"Anomaly Type: {result['anomaly_type']}")
    print(f"Confidence: {result['confidence']:.2f}")
    print("\nRemediation Steps:")
    for i, step in enumerate(result['remediation_steps'], 1):
        print(f"{i}. {step.get('action', 'Unknown action')}: {step.get('description', 'No description')}")
    print("\nExplanation:")
    print(result['explanation'])

if __name__ == "__main__":
    main() 