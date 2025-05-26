# imports
import os
import goodfire
import argparse
from dotenv import load_dotenv
from lm_eval.models.openai_completions import OpenAIChatCompletion
from lm_eval import evaluator
import tabulate
from lm_eval.api.registry import register_model
from lm_eval.api.registry import get_model
from datetime import datetime

# the code we need to interface with lm-eval-harness
@register_model("goodfire-chat-bench")
class GoodfireChatCompletion(OpenAIChatCompletion):
    def __init__(self, model=None, base_url=None, **kwargs):
        super().__init__(model=model, base_url=base_url, **kwargs)
        self.model_args = kwargs
        self.sample_count = 0

    def _create_payload(self, messages, generate=False, gen_kwargs=None, seed=1234, eos=None, **kwargs):
        payload = super()._create_payload(messages, generate, gen_kwargs, seed, eos, **kwargs)
        
        if "controller" in self.model_args:
            payload["controller"] = self.model_args["controller"]
        
        if generate:  # Only increment for actual completions, not logprobs
            self.sample_count += 1
            print(f"Processing sample {self.sample_count}...")
            payload["stop"] = ["</s>", "<|im_end|>", "<|endoftext|>"]
        
        return payload

# the ascii title that loads when this file is run
def print_ascii_title():
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

# loading and returning the goodfire client
def load_client():
    load_dotenv()
    GOODFIRE_API_KEY = os.getenv("GOODFIRE_API_KEY")
    os.environ["OPENAI_API_KEY"] = GOODFIRE_API_KEY
    if GOODFIRE_API_KEY is None:
        raise ValueError("GOODFIRE_API_KEY is not set")
    else:
        print("GOODFIRE_API_KEY is set")
    client = goodfire.Client(api_key=GOODFIRE_API_KEY)
    return client

# loading and returning the goodfire variant, with option for 70b or 8b
def get_variant(model):
    if model == "70b":
        return goodfire.Variant("meta-llama/Llama-3.3-70B-Instruct")
    elif model == "8b":
        return goodfire.Variant("meta-llama/Meta-Llama-3.1-8B-Instruct")
    # elif model == "r1":
    #     return goodfire.Variant("deepseek-ai/DeepSeek-R1")
    else:
        raise ValueError(f"Invalid model: {model}")

# loading and returning the system prompt for the benchmark
def get_system_prompt(benchmark):
    if benchmark == "truthfulqa_gen":
        return """
        You are an assistant helping to provide direct answers to questions. Answer with a single,
        clear sentence (unless explicitly asked for more detail)
        """
    
    elif benchmark == "gsm8k":
        return """
        You are an assistant helping to solve math problems. When answering:
        1. At the end of your response, you MUST put your final answer in the format '#### <final numerical answer>'
        2. Do not include units or symbols in the final answer line
        3. Only include the number after the ####
        4. You MUST put your final answer after the ####

        Example:
        Question: If John has 3 apples and buys 2 more, how many does he have?
        Answer: John starts with 3 apples.
        He buys 2 more apples.
        So he has 3 + 2 = 5 apples total.
        #### 5

        Example:
        Question: If John spends $5 on lunch and $3 on coffee, his total is:
        $5 + $3 = $8 total spent
        #### 8

        As you can see from the examples, the correct format is putting your final numerical answer after the #### in the format #### <final numerical answer>. 
        """
    else:
        raise ValueError(f"System prompt currently not defined for benchmark: {benchmark} (this benchmark is not yet supported)")

# querying the goodfire api for the topk features
def find_features(input_variant, prompt, topk, client):
    return client.features.search(f"{prompt}",model=input_variant, top_k=topk)

# applying the features to the variant
def apply_features(input_variant, features, strength):
    for i, feature in enumerate(features):
        input_variant.set(features[i], strength)
    return input_variant

# accessing the controller settings for the variant, with option to return as a string or json object
def get_controller_settings(input_variant, return_string=True):
    if return_string:
        return str(input_variant.controller.json())
    else:
        return input_variant.controller.json()

# verifying that our custom 'goodfire-chat-bench' model is properly registered
def verify_model_registration():
    try:
        model_class = get_model("goodfire-chat-bench")
        print(f"✅ Model 'goodfire-chat-bench' is registered: {model_class}")
        return True
    except Exception as e:
        print(f"❌ Model registration failed: {e}")
        return False

# defining the model evaluation parameters, with option for temperature
def define_model_eval(input_variant, temperature):
    eval_model = {
        "model": "goodfire-chat-bench",
        "model_args": {
            "model": input_variant.base_model,
            "base_url": "https://api.goodfire.ai/api/inference/v1/chat/completions",
            "temperature": temperature,
            "eos_string": "</s>",
            "controller": get_controller_settings(input_variant, return_string=False)
        }
    }
    return eval_model

# running the evaluation, with option for limit and task
def run_eval(defined_model_eval, task, limit, system_prompt, seed=1234):
    eval_results = evaluator.simple_evaluate(
        model = defined_model_eval["model"],
        model_args = defined_model_eval["model_args"],
        tasks = [task],
        limit = limit,
        # bootstrap_iters = 10,
        apply_chat_template = True,
        system_instruction = system_prompt,
        random_seed = seed,  # Ensure consistent question selection
    )
    return eval_results

# formatting the controller settings in a nice table format
def format_controller_settings(input_variant):
    controller_json = get_controller_settings(input_variant, return_string=False)
    
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

# formatting the evaluation results
def format_results(eval_results, baseline_results=None):
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

# creating the results directory if it doesn't exist
def ensure_results_directory():
    if not os.path.exists("results"):
        os.makedirs("results")
        print("📁 Created results/ directory")

# generating a filename for the log
def generate_log_filename(log_name=None):
    if log_name:
        return f"results/{log_name}.txt"
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"results/benchmark_{timestamp}.txt"

# saving results to a log file
def save_results_to_file(filename, model, prompt, benchmark, topk, strength, limit, temperature, 
                        variant, eval_results, baseline_variant=None, baseline_results=None, raw_results=False):
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

def main_thread(model, prompt, benchmark = "gsm8k",topk=5, strength=0.3, limit=5, use_baseline=False, temperature=0.0, raw_results=False, log_name=None):
    print_ascii_title()
    print(f"\n🚀 Starting benchmark with model: {model}")
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
        log_filename, model, prompt, benchmark, topk, strength, 
        limit, temperature, variant, eval_results, baseline_variant, baseline_results, raw_results
    )
    
    print("🎉 Success -- benchmark complete!")
    print(f"✅ Results saved to {log_filename}")

# parsing the command line arguments
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Goodfire benchmark')
    parser.add_argument('--model', type=str, default='70b', help='Model to use ("70b" or "8b")')
    parser.add_argument('--prompt', type=str, default="mathematical reasoning and problem solving", help='Prompt to search for autointerp feature labels')
    parser.add_argument('--task', type=str, default='gsm8k', help='Benchmark to run ("truthfulqa_gen" or "gsm8k")')
    parser.add_argument('--topk', type=int, default=5, help='Number of features to use')
    parser.add_argument('--strength', type=float, default=0.3, help='Feature strength')
    parser.add_argument('--limit', type=int, default=5, help='Number of samples to run')
    parser.add_argument('--baseline', type=bool, default=False, help='Use baseline model')
    parser.add_argument('--temp', type=float, default=0.0, help='Temperature')
    parser.add_argument('--raw_results', type=bool, default=False, help='Print raw results')
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