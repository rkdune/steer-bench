"""
Evaluation functions for the SAE Steer Benchmark.
Includes model setup, system prompts, and evaluation execution.
"""

import os
import goodfire
from dotenv import load_dotenv
from lm_eval import evaluator
from lm_eval.api.registry import get_model


def load_client():
    """Load and return the Goodfire client."""
    load_dotenv()
    GOODFIRE_API_KEY = os.getenv("GOODFIRE_API_KEY")
    os.environ["OPENAI_API_KEY"] = GOODFIRE_API_KEY
    if GOODFIRE_API_KEY is None:
        raise ValueError(
            "GOODFIRE_API_KEY is not set. Please set it using one of these methods:\n"
            "1. Create a .env file with: GOODFIRE_API_KEY=your_api_key_here\n"
            "2. Set environment variable: export GOODFIRE_API_KEY=your_api_key_here\n"
            "3. Run with: GOODFIRE_API_KEY=your_api_key_here python bench.py [args]\n"
            "See README.md for detailed instructions."
        )
    else:
        print("GOODFIRE_API_KEY is set")
    client = goodfire.Client(api_key=GOODFIRE_API_KEY)
    return client


def get_variant(model):
    """Load and return the Goodfire variant, with option for 70b or 8b."""
    if model == "70b":
        return goodfire.Variant("meta-llama/Llama-3.3-70B-Instruct")
    elif model == "8b":
        return goodfire.Variant("meta-llama/Meta-Llama-3.1-8B-Instruct")
    # elif model == "r1":
    #     return goodfire.Variant("deepseek-ai/DeepSeek-R1")
    else:
        raise ValueError(f"Invalid model: {model}")


def get_system_prompt(benchmark):
    """Load and return the system prompt for the benchmark."""
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


def find_features(input_variant, prompt, topk, client):
    """Query the Goodfire API for the top-k features."""
    return client.features.search(f"{prompt}", model=input_variant, top_k=topk)


def apply_features(input_variant, features, strength):
    """Apply the features to the variant."""
    for i, feature in enumerate(features):
        input_variant.set(features[i], strength)
    return input_variant


def get_controller_settings(input_variant, return_string=True):
    """Access the controller settings for the variant, with option to return as a string or json object."""
    if return_string:
        return str(input_variant.controller.json())
    else:
        return input_variant.controller.json()


def verify_model_registration():
    """Verify that our custom 'goodfire-chat-bench' model is properly registered."""
    try:
        model_class = get_model("goodfire-chat-bench")
        print(f"✅ Model 'goodfire-chat-bench' is registered: {model_class}")
        return True
    except Exception as e:
        print(f"❌ Model registration failed: {e}")
        return False


def define_model_eval(input_variant, temperature):
    """Define the model evaluation parameters, with option for temperature."""
    # Determine appropriate eos_string based on model
    model_name = input_variant.base_model.lower()
    if "llama" in model_name:
        eos_string = "</s>"
    elif "gpt" in model_name:
        eos_string = "<|endoftext|>"
    else:
        # Default fallback
        eos_string = "</s>"
    
    eval_model = {
        "model": "goodfire-chat-bench",
        "model_args": {
            "model": input_variant.base_model,
            "base_url": "https://api.goodfire.ai/api/inference/v1/chat/completions",
            "temperature": temperature,
            "eos_string": eos_string,
            "controller": get_controller_settings(input_variant, return_string=False)
        }
    }
    return eval_model


def run_eval(defined_model_eval, task, limit, system_prompt, seed=1234):
    """Run the evaluation, with option for limit and task."""
    eval_results = evaluator.simple_evaluate(
        model=defined_model_eval["model"],
        model_args=defined_model_eval["model_args"],
        tasks=[task],
        limit=limit,
        bootstrap_iters = 100000,  # Use proper default for statistical significance
        apply_chat_template=True,
        system_instruction=system_prompt,
        random_seed=seed,  # Ensure consistent question selection
        cache_requests=False
    )
    print("Bootstrap iterations config:", eval_results.get('config', {}).get('bootstrap_iters'))
    return eval_results