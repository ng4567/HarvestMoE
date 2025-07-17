import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from collections import defaultdict
import json

MODEL_NAME = "microsoft/Phi-3.5-MoE-instruct"
MODEL_REVISION = "c5ec1449e5376ad4c7031bf0d51eabf5e7d08887"

class PhiMoE:
    def __init__(self):
        print("Loading model + tokenizer into GPU memory...\n")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, revision=MODEL_REVISION, device_map="auto", torch_dtype=torch.float16
        )
        # Dictionary to store expert activations
        self.expert_activations = defaultdict(set)
        self.expert_activation_freq = defaultdict(int)
        
        self.register_expert_hooks(self.model, self.expert_activations)

    def register_expert_hooks(self, model, expert_activations):
        # Function to register hooks for capturing expert activations 
        def hook_fn(module, input, output, layer_idx):
            # The gate output contains routing probabilities
            # For PhiMoE, the output is just the router logits directly
            gate_output = output if isinstance(output, torch.Tensor) else output[0]
            
            # Get top-2 experts (PhiMoE uses top-2 routing)
            _, top_experts = torch.topk(gate_output, k=2, dim=-1)
            
    
            # Convert to numpy and iterate through all tokens to add expert indices to set
            top_experts_np = top_experts.detach().cpu().numpy()
            
            # Handle different tensor shapes
            if len(top_experts_np.shape) == 3:
                # Shape: [batch_size, seq_len, k]
                batch_size, seq_len, k = top_experts_np.shape
                for batch_idx in range(batch_size):
                    for token_idx in range(seq_len):
                        for expert_rank in range(k):
                            expert_idx = top_experts_np[batch_idx, token_idx, expert_rank]
                            expert_activations[f"layer_{layer_idx}"].add(int(expert_idx))
            elif len(top_experts_np.shape) == 2:
                # Shape: [seq_len, k] (no batch dimension)
                seq_len, k = top_experts_np.shape
                for token_idx in range(seq_len):
                    for expert_rank in range(k):
                        expert_idx = top_experts_np[token_idx, expert_rank]
                        expert_activations[f"layer_{layer_idx}"].add(int(expert_idx))
            else:
                # Handle other shapes by flattening
                for expert_idx in top_experts_np.flatten():
                    expert_activations[f"layer_{layer_idx}"].add(int(expert_idx))
        
        # Register hooks for each MoE layer
        for i, layer in enumerate(model.model.layers):
            layer.block_sparse_moe.gate.register_forward_hook(
                lambda module, input, output, layer_idx=i: hook_fn(module, input, output, layer_idx)
            )

    def forward(self, text):
        # Clear previous activations
        inputs = self.tokenizer(text, return_tensors="pt").to("cuda:1")
        with torch.no_grad():
            _ = self.model(**inputs)

    def get_expert_activations(self):
        return self.expert_activations

# Example usage:
if __name__ == "__main__":
    # Initialize the model
    phi_moe = PhiMoE()
    
    # Run inference on some text
    with open("/home/azureuser/moe-lightning-fork-new/benchmarks/mtbench/question.jsonl", "r") as f:
        for line in f:
            item = json.loads(line.strip())
            # Each item has a "turns" array with questions
            for turn_idx, question in enumerate(item["turns"]):
                print(f"Processing question {item['question_id']}, turn {turn_idx + 1}: {question[:50]}...")
                phi_moe.forward(question)
    
    
    # Get the expert activations
    activations = phi_moe.get_expert_activations()
    
    total = set(range(16))

    # Print results
    for layer, experts in activations.items():
        print(f"{layer}: activated experts {sorted(list(experts))}")
        print(f"{layer}: unactivated experts {sorted(list(total - experts))}")
