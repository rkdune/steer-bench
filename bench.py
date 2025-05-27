"""
Main benchmark script for SAE Steer Benchmark.
This script orchestrates the evaluation of Goodfire models with feature steering.
"""

import argparse

# Import from our custom modules
from goodfire_model import GoodfireChatCompletion  # This registers the model
from utils import (
    print_ascii_title,
    format_controller_settings,
    format_results,
    ensure_results_directory,
    generate_log_filename,
    save_results_to_file
)
from evaluation import (
    load_client,
    get_variant,
    get_system_prompt,
    find_features,
    apply_features,
    verify_model_registration,
    define_model_eval,
    run_eval
)


def main_thread(model, prompt, benchmark, topk, strength, limit, use_baseline, temperature, raw_results, log_name):
    """Main execution thread for the benchmark."""
    print_ascii_title()

    if model == "70b":
        full_model = "Llama-3.3-70B-Instruct"
    elif model == "8b":
        full_model = "Meta-Llama-3.1-8B-Instruct"
    else:
        full_model = model
    print(f"\n🚀 Starting benchmark with model: {full_model}")
    print(f"📝 Using prompt: {prompt}")
    print(f"🎯 Benchmark: {benchmark}")
    print(f"⚙️  Configuration: topk={topk}, strength={strength}, limit={limit}")
    if use_baseline:
        print(f"📊 Baseline comparison: ENABLED")
    print()
    
    # Ensure results directory exists and generate filename
    ensure_results_directory()
    log_filename = generate_log_filename(log_name)
    print(f"📝 Results will be saved to: {log_filename}\n")
    
    print("🔄 Loading client...")
    client = load_client()
    print("✅ Client loaded\n")

    print("🔄 Initializing model variant...")
    variant = get_variant(model)
    print("✅ Model variant initialized\n")
    
    # Create baseline variant if needed
    baseline_variant = None
    baseline_results = None
    if use_baseline:
        print("🔄 Initializing baseline variant...")
        baseline_variant = get_variant(model)
        print("✅ Baseline variant initialized\n")
    
    print(f"🔍 Searching for top {topk} features...")
    features = find_features(variant, prompt, topk, client)
    print(f"✅ Found {len(features)} features\n")
    
    print("⚡ Applying features to steered variant...")
    variant = apply_features(variant, features, strength)
    print("✅ Features applied to steered variant\n")
    
    print("📊 Steered variant controller settings:")
    print(format_controller_settings(variant))
    
    if use_baseline:
        print("\n📊 Baseline variant controller settings:")
        print(format_controller_settings(baseline_variant))
    
    print("\n✨ Setup complete! Ready to run benchmark.")

    print("🔄 Getting system prompt...")
    system_prompt = get_system_prompt(benchmark)
    print("✅ System prompt loaded\n")

    print("🔄 Verifying model registration...")
    if not verify_model_registration():
        print("❌ Model registration verification failed. Exiting.")
        return
    print("✅ Model registration verified\n")

    print("🔄 Defining steered model evaluation parameters...")
    defined_model_eval = define_model_eval(variant, temperature)
    print("✅ Steered model evaluation parameters defined\n")

    print("🚀 Running steered evaluation...")
    eval_seed = 1234  # Use consistent seed for fair comparison
    eval_results = run_eval(defined_model_eval, benchmark, limit, system_prompt, eval_seed)
    print("✅ Steered evaluation complete\n")

    # Run baseline evaluation if requested
    if use_baseline:
        print("🔄 Defining baseline model evaluation parameters...")
        baseline_model_eval = define_model_eval(baseline_variant, temperature)
        print("✅ Baseline model evaluation parameters defined\n")
        
        print("🚀 Running baseline evaluation...")
        # Use the same seed to ensure same questions are evaluated
        baseline_results = run_eval(baseline_model_eval, benchmark, limit, system_prompt, eval_seed)
        print("✅ Baseline evaluation complete\n")

    print("🔍 Printing results...")
    print(format_results(eval_results, baseline_results))

    if raw_results:
        print("🔍 Printing raw steered eval results...")
        print(eval_results)
        if use_baseline:
            print("🔍 Printing raw baseline eval results...")
            print(baseline_results)
    else:
        print("🔍 Raw results not requested. Skipping...")
    
    # Save results to file
    print(f"💾 Saving results to {log_filename}...")
    save_results_to_file(
        log_filename, full_model, prompt, benchmark, topk, strength, 
        limit, temperature, variant, eval_results, baseline_variant, baseline_results, raw_results
    )
    
    print("🎉 Success -- benchmark complete!")
    print(f"✅ Results saved to {log_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Goodfire benchmark')
    parser.add_argument('--model', type=str, default='70b', help='Model to use ("70b" or "8b")')
    parser.add_argument('--prompt', type=str, default="mathematical reasoning and problem solving", help='Prompt to search for autointerp feature labels')
    parser.add_argument('--task', type=str, default='gsm8k', help='Benchmark to run ("truthfulqa_gen" or "gsm8k")')
    parser.add_argument('--topk', type=int, default=2, help='Number of features to use')
    parser.add_argument('--strength', type=float, default=0.3, help='Feature strength')
    parser.add_argument('--limit', type=int, default=5, help='Number of samples to run')
    parser.add_argument('--baseline', action='store_true', help='Use baseline model comparison')
    parser.add_argument('--temp', type=float, default=0.0, help='Temperature')
    parser.add_argument('--raw_results', action='store_true', help='Print raw results')
    parser.add_argument('--name', type=str, default=None, help='Custom name for the log file (without .txt extension)')

    args = parser.parse_args()
    
    main_thread(
        model=args.model,
        prompt=args.prompt,
        benchmark=args.task,
        topk=args.topk,
        strength=args.strength,
        limit=args.limit,
        use_baseline=args.baseline,
        temperature=args.temp,
        raw_results=args.raw_results,
        log_name=args.name
    )