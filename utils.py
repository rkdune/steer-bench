"""
Utility functions for the SAE Steer Benchmark.
Includes printing, formatting, and file operations.
"""

import os
import tabulate
from datetime import datetime


def print_ascii_title():
    """Print the ASCII title banner for the benchmark."""
    # standard font from https://patorjk.com/software/taag/#p=testall&f=3D%20Diagonal&t=SAE%20Steer%20Bench
    ascii_title = r"""

   _____         ______    _____ _                   ____                  _     
  / ____|  /\   |  ____|  / ____| |                 |  _ \                | |    
 | (___   /  \  | |__    | (___ | |_ ___  ___ _ __  | |_) | ___ _ __   ___| |__  
  \___ \ / /\ \ |  __|    \___ \| __/ _ \/ _ \ '__| |  _ < / _ \ '_ \ / __| '_ \ 
  ____) / ____ \| |____   ____) | ||  __/  __/ |    | |_) |  __/ | | | (__| | | |
 |_____/_/    \_\______| |_____/ \__\___|\___|_|    |____/ \___|_| |_|\___|_| |_|
                                                                                 
                                                                                 
       
    """
    print(ascii_title)


def format_controller_settings(input_variant):
    """Format the controller settings in a nice table format."""
    controller_json = input_variant.controller.json()
    
    # Extract feature information from the controller
    controller_data = [["Feature Name", "Strength"]]
    
    if 'interventions' in controller_json:
        for intervention in controller_json['interventions']:
            if 'features' in intervention and 'features' in intervention['features']:
                for feature in intervention['features']['features']:
                    feature_name = feature.get('label', 'Unknown Feature')
                    strength = intervention.get('value', 0.0)
                    controller_data.append([feature_name, f"{strength:.1f}"])
    
    return tabulate.tabulate(controller_data, headers="firstrow", tablefmt="grid")


def format_results(eval_results, baseline_results=None):
    """Format the evaluation results in a nice table format."""
    # Get the benchmark name from the results (first key in results dict)
    benchmark_name = list(eval_results['results'].keys())[0]
    benchmark_metrics = eval_results['results'][benchmark_name]
    
    # helper function to format numeric values based on benchmark type
    def format_value(value, benchmark_name):
        try:
            # Convert to float if it's a string
            if isinstance(value, str):
                float_value = float(value)
            else:
                float_value = value
            
            # For gsm8k, format as percentage; for truthfulqa_gen, keep as decimal
            if benchmark_name == "gsm8k":
                return f"{float_value:.1%}"
            else:
                return f"{float_value:.3f}"
        except (ValueError, TypeError):
            # If conversion fails, return the value as-is
            return str(value)
    
    if baseline_results is None:
        # Original single variant format
        results_data = [["Metric", "Variant"]]
        
        # Add all available metrics dynamically
        for metric_name, metric_value in benchmark_metrics.items():
            # Format metric name for display
            display_name = metric_name.replace('_', ' ').replace(',', ' ').title()
            results_data.append([display_name, format_value(metric_value, benchmark_name)])
            
    else:
        # Comparison format with both steered and baseline
        baseline_benchmark_metrics = baseline_results['results'][benchmark_name]
        results_data = [["Metric", "Steered", "Baseline"]]
        
        # Add all available metrics dynamically
        for metric_name, metric_value in benchmark_metrics.items():
            if metric_name in baseline_benchmark_metrics:
                display_name = metric_name.replace('_', ' ').replace(',', ' ').title()
                results_data.append([
                    display_name,
                    format_value(metric_value, benchmark_name),
                    format_value(baseline_benchmark_metrics[metric_name], benchmark_name)
                ])
    
    return tabulate.tabulate(results_data, headers="firstrow", tablefmt="grid")


def ensure_results_directory():
    """Create the results directory if it doesn't exist."""
    if not os.path.exists("results"):
        os.makedirs("results")
        print("📁 Created results/ directory")


def generate_log_filename(log_name=None):
    """Generate a filename for the log file."""
    if log_name:
        return f"results/{log_name}.txt"
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"results/benchmark_{timestamp}.txt"


def save_results_to_file(filename, model, prompt, benchmark, topk, strength, limit, temperature, 
                        variant, eval_results, baseline_variant=None, baseline_results=None, raw_results=False):
    """Save benchmark results to a log file."""
    with open(filename, 'w', encoding='utf-8') as f:
        # Write header with timestamp
        f.write("=" * 80 + "\n")
        f.write("SAE STEER BENCHMARK RESULTS\n")
        f.write("=" * 80 + "\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Write configuration
        f.write("CONFIGURATION:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Model: {model}\n")
        f.write(f"Prompt: {prompt}\n")
        f.write(f"Benchmark: {benchmark}\n")
        f.write(f"Top-K Features: {topk}\n")
        f.write(f"Feature Strength: {strength}\n")
        f.write(f"Sample Limit: {limit}\n")
        f.write(f"Temperature: {temperature}\n")
        f.write(f"Baseline Comparison: {'Yes' if baseline_results else 'No'}\n\n")
        
        # Write controller settings table (reuse formatting function)
        f.write("STEERED VARIANT CONTROLLER SETTINGS:\n")
        f.write("-" * 40 + "\n")
        controller_table = format_controller_settings(variant)
        f.write(controller_table + "\n\n")
        
        # Write baseline controller settings if available
        if baseline_variant:
            f.write("BASELINE VARIANT CONTROLLER SETTINGS:\n")
            f.write("-" * 40 + "\n")
            baseline_controller_table = format_controller_settings(baseline_variant)
            f.write(baseline_controller_table + "\n\n")
        
        # Write results table (reuse formatting function)
        f.write("EVALUATION RESULTS:\n")
        f.write("-" * 40 + "\n")
        results_table = format_results(eval_results, baseline_results)
        f.write(results_table + "\n\n")
        
        f.write("RAW STEERED EVALUATION RESULTS:\n")
        f.write("-" * 40 + "\n")
        f.write(str(eval_results) + "\n\n")
        
        # Write baseline raw results if available
        if baseline_results:
            f.write("RAW BASELINE EVALUATION RESULTS:\n")
            f.write("-" * 40 + "\n")
            f.write(str(baseline_results) + "\n") 