"""
Custom Goodfire model class for lm-eval-harness integration.
"""

from lm_eval.models.openai_completions import OpenAIChatCompletion
from lm_eval.api.registry import register_model


@register_model("goodfire-chat-bench")
class GoodfireChatCompletion(OpenAIChatCompletion):
    """Custom Goodfire model class that extends OpenAI chat completion for lm-eval-harness."""
    
    def __init__(self, model=None, base_url=None, **kwargs):
        super().__init__(model=model, base_url=base_url, **kwargs)
        self.model_args = kwargs
        self.sample_count = 0

    def _create_payload(self, messages, generate=False, gen_kwargs=None, seed=1234, eos=None, **kwargs):
        payload = super()._create_payload(messages, generate, gen_kwargs, seed, eos, **kwargs)
        
        # Handle controller - provide proper empty structure if none provided
        if "controller" in self.model_args:
            controller = self.model_args["controller"]
            # If controller is empty or missing required fields, provide proper empty structure
            if not controller or "scopes" not in controller or "interventions" not in controller:
                payload["controller"] = {
                    "interventions": [],
                    "scopes": [],
                    "name": "empty_controller",
                    "nonzero_strength_threshold": None,
                    "min_nudge_entropy": None,
                    "max_nudge_entropy": None
                }
            else:
                payload["controller"] = controller
        
        if generate:  # Only increment for actual completions, not logprobs
            self.sample_count += 1
            print(f"Processing sample {self.sample_count}...")

            # Only add default stop tokens if none are provided
            if "stop" not in payload or not payload["stop"]:
                # Use model-specific stop tokens based on the model name
                if "llama" in self.model_args.get("model", "").lower():
                    payload["stop"] = ["</s>", "<|im_end|>", "<|endoftext|>"]
                else:
                    # Default stop tokens for other models
                    payload["stop"] = ["<|endoftext|>", "<|im_end|>"]
        
        return payload 