import os
import glob
import matplotlib.pyplot as plt
import numpy as np

# Configuration
output_dir = r"EdgeCloudSim\scripts\sample_app1\output\ite1"
algorithms = ["ECO_EDGE_FIT", "NEXT_FIT", "RANDOM_FIT", "DQN_FIT"]
labels = {
    "ECO_EDGE_FIT": "Eco-Edge",
    "NEXT_FIT": "Next-Fit (Heuristic)",
    "RANDOM_FIT": "Random-Fit",
    "DQN_FIT": "DQN Baseline",
}
colors = {
    "ECO_EDGE_FIT": "green",
    "NEXT_FIT": "blue",
    "RANDOM_FIT": "red",
    "DQN_FIT": "darkorange",
}
markers = {"ECO_EDGE_FIT": "o", "NEXT_FIT": "s", "RANDOM_FIT": "^", "DQN_FIT": "D"}
devices = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

def parse_logs():
    data = {alg: {"latency": [], "success_rate": [], "energy": [], "failed_capacity": [], "completed": []} for alg in algorithms}
    
    for alg in algorithms:
        for dev in devices:
            file_pattern = f"SIMRESULT_TWO_TIER_WITH_EO_{alg}_{dev}DEVICES_ALL_APPS_GENERIC.log"
            filepath = os.path.join(output_dir, file_pattern)
            
            if not os.path.exists(filepath):
                print(f"File not found: {filepath}")
                # Append NaNs if not found
                data[alg]["latency"].append(np.nan)
                data[alg]["success_rate"].append(np.nan)
                data[alg]["energy"].append(np.nan)
                data[alg]["failed_capacity"].append(np.nan)
                data[alg]["completed"].append(0)
                continue
                
            with open(filepath, 'r') as f:
                lines = f.readlines()
                
            # Filter out the header if present
            content_lines = [l.strip() for l in lines if not l.startswith('#') and len(l.strip()) > 0]
            
            if len(content_lines) >= 6:
                # Delimiter is usually ';' in EdgeCloudSim
                delim = ';' if ';' in content_lines[0] else '\t'
                
                gen_stats = content_lines[0].split(delim)
                completed = float(gen_stats[0])
                failed = float(gen_stats[1])
                uncompleted = float(gen_stats[2])
                service_time = float(gen_stats[4])
                failed_cap = float(gen_stats[9])
                
                total_tasks = completed + failed + uncompleted
                success_rate = (completed / total_tasks) * 100 if total_tasks > 0 else 0
                
                perf_stats = content_lines[5].split(delim)
                thinking_energy = float(perf_stats[2]) if len(perf_stats) > 2 else 0.0
                
                data[alg]["latency"].append(service_time)
                data[alg]["success_rate"].append(success_rate)
                data[alg]["energy"].append(thinking_energy)
                data[alg]["failed_capacity"].append(failed_cap)
                data[alg]["completed"].append(completed)
    
    return data

def plot_graphs(data):
    # 1. Latency (Service Time) vs Number of Devices
    plt.figure(figsize=(8, 6))
    for alg in algorithms:
        plt.plot(devices, data[alg]["latency"], label=labels[alg], color=colors[alg], marker=markers[alg], linewidth=2.5, markersize=8)
    plt.xlabel('Number of Edge Devices (Scale)', fontsize=14)
    plt.ylabel('Average Service Time (Seconds)', fontsize=14)
    plt.title('End-to-End Latency vs. Network Scale', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig('fig_scalability_latency.pdf')
    plt.close()

    # 2. Task Success Rate vs Number of Devices
    plt.figure(figsize=(8, 6))
    for alg in algorithms:
        plt.plot(devices, data[alg]["success_rate"], label=labels[alg], color=colors[alg], marker=markers[alg], linewidth=2.5, markersize=8)
    plt.xlabel('Number of Edge Devices (Scale)', fontsize=14)
    plt.ylabel('Task Success Rate (%)', fontsize=14)
    plt.title('Task Completion Rate under Heavy Load', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12, loc='lower left')
    plt.tight_layout()
    plt.savefig('fig_success_rate.pdf')
    plt.close()

    # 3. Failures due to VM Capacity vs Number of Devices
    plt.figure(figsize=(8, 6))
    for alg in algorithms:
        plt.plot(devices, data[alg]["failed_capacity"], label=labels[alg], color=colors[alg], marker=markers[alg], linewidth=2.5, markersize=8)
    plt.xlabel('Number of Edge Devices (Scale)', fontsize=14)
    plt.ylabel('Failed Tasks (VM Overload)', fontsize=14)
    plt.title('Orchestration Robustness to Congestion', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig('fig_failed_capacity.pdf')
    plt.close()

    # 4. Total Thinking Energy vs Number of Devices
    plt.figure(figsize=(8, 6))
    for alg in algorithms:
        plt.plot(devices, data[alg]["energy"], label=labels[alg], color=colors[alg], marker=markers[alg], linewidth=2.5, markersize=8)
    plt.xlabel('Number of Edge Devices (Scale)', fontsize=14)
    plt.ylabel('SLM Inference Energy (Joules)', fontsize=14)
    plt.title('Energy Overhead of Decentralized Intelligence', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig('fig_energy_overhead.pdf')
    plt.close()

if __name__ == "__main__":
    print("Parsing EdgeCloudSim logs...")
    data = parse_logs()
    print("Generating Matplotlib plots for the paper...")
    plot_graphs(data)
    print("Done! High-resolution PDFs have been generated.")
