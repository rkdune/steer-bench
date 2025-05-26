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
def get_variant(model="70b"):
    if model == "70b":
        return goodfire.Variant("meta-llama/Llama-3.3-70B-Instruct")
    elif model == "8b":
        return goodfire.Variant("meta-llama/Llama-3.3-8B-Instruct")
    else:
        raise ValueError(f"Invalid model: {model}")

# loading and returning the system prompt for the benchmark
def get_system_prompt(benchmark):
    if benchmark == "truthfulqa":
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
        raise ValueError(f"System prompt currently not defined for benchmark: {benchmark}")

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
def run_eval(defined_model_eval, task, limit, system_prompt):
    eval_results = evaluator.simple_evaluate(
        model = defined_model_eval["model"],
        model_args = defined_model_eval["model_args"],
        tasks = [task],
        limit = limit,
        # bootstrap_iters = 10,
        apply_chat_template = True,
        system_instruction = system_prompt,
    )
    return eval_results

# printing the controller settings in a nice table format
def print_controller_settings(input_variant):
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
    
    print(tabulate.tabulate(controller_data, headers="firstrow", tablefmt="grid"))

# printing the evaluation results
def print_results(eval_results, use_baseline=False):
    results_data = [    
        ["Metric", "Variant"],
        ["Strict Match", f"{eval_results['results']['gsm8k']['exact_match,strict-match']:.1%}"],
        ["Strict Stderr", f"{eval_results['results']['gsm8k']['exact_match_stderr,strict-match']:.1%}"],
        ["Flexible Match", f"{eval_results['results']['gsm8k']['exact_match,flexible-extract']:.1%}"],
        ["Flexible Stderr", f"{eval_results['results']['gsm8k']['exact_match_stderr,flexible-extract']:.1%}"]
    ]
    print(tabulate.tabulate(results_data, headers="firstrow", tablefmt="grid"))

def main_thread(model, prompt, benchmark = "gsm8k",topk=5, strength=0.3, limit=5, use_baseline=False, temperature=0.0, raw_results=False):
    print_ascii_title()
    print(f"\n🚀 Starting benchmark with model: {model}")
    print(f"📝 Using prompt: {prompt}")
    print(f"🎯 Benchmark: {benchmark}")
    print(f"⚙️ Configuration: topk={topk}, strength={strength}, limit={limit}\n")
    
    print("🔄 Loading client...")
    client = load_client()
    print("✅ Client loaded\n")

    print("🔄 Initializing model variant...")
    variant = get_variant(model)
    print("✅ Model variant initialized\n")
    
    print(f"🔍 Searching for top {topk} features...")
    features = find_features(variant, prompt, topk, client)
    print(f"✅ Found {len(features)} features\n")
    
    print("⚡ Applying features to variant...")
    variant = apply_features(variant, features, strength)
    print("✅ Features applied\n")
    
    print("📊 Controller settings:")
    print_controller_settings(variant)
    print("\n✨ Setup complete! Ready to run benchmark.")

    
    print("🔄 Getting system prompt...")
    system_prompt = get_system_prompt(benchmark)
    print("✅ System prompt loaded\n")

    print("🔄 Verifying model registration...")
    if not verify_model_registration():
        print("❌ Model registration verification failed. Exiting.")
        return
    print("✅ Model registration verified\n")

    print("🔄 Defining model evaluation parameters...")
    defined_model_eval = define_model_eval(variant, temperature)
    print("✅ Model evaluation parameters defined\n")

    print("🚀 Running evaluation...")
    eval_results = run_eval(defined_model_eval, benchmark, limit, system_prompt)
    print("✅ Evaluation complete\n")

    print("🔍 Printing results...")
    print_results(eval_results, use_baseline)

    if raw_results:
        print("🔍 Printing raw eval results...")
        print(eval_results)
    else:
        print("🔍 Raw results not requested. Skipping...")
    
    print("🎉 Success -- benchmark complete!")

# parsing the command line arguments
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Goodfire benchmark')
    parser.add_argument('--model', type=str, default='70b', help='Model to use ("70b" or "8b")')
    parser.add_argument('--prompt', type=str, default="mathematical reasoning and problem solving", help='Prompt to use')
    parser.add_argument('--task', type=str, default='gsm8k', help='Benchmark to run ("truthfulqa" or "gsm8k")')
    parser.add_argument('--topk', type=int, default=5, help='Number of features to use')
    parser.add_argument('--strength', type=float, default=0.3, help='Feature strength')
    parser.add_argument('--limit', type=int, default=5, help='Number of samples to run')
    parser.add_argument('--use_baseline', type=bool, default=False, help='Use baseline model')
    parser.add_argument('--temp', type=float, default=0.0, help='Temperature')
    parser.add_argument('--raw_results', type=bool, default=False, help='Print raw results')

    args = parser.parse_args()
    
    main_thread(
        model=args.model,
        prompt=args.prompt,
        benchmark=args.task,
        topk=args.topk,
        strength=args.strength,
        limit=args.limit,
        use_baseline=args.use_baseline,
        temperature=args.temp,
        raw_results=args.raw_results
    )
