import torch
# We import our structural tissue from our first file
from brain_tissue import LocalNeuralLayer, build_local_child_brain

class CognitiveSenses:
    """
    👁️ Gives our blank-slate AI the ability to process sensory information 
    by passing signals forward through its neural matrix.
    """
    @staticmethod
    def transmit_signal(input_data: torch.Tensor, layer: LocalNeuralLayer) -> torch.Tensor:
        """
        ⚡ Computes the Forward Pass: Signal = ReLU( (Input x Weights) + Bias )
        """
        # Step A: Perform matrix multiplication of inputs and weights, then add bias
        # We use layer.weights.t() to transpose the matrix dimensions for clean multiplication alignment
        raw_electrical_signal = torch.matmul(input_data, layer.weights.t()) + layer.biases
        
        # Step B: The Synaptic Trigger (ReLU Activation Function)
        # It clips all negative numbers to 0, matching how biological neurons either fire or remain silent
        activated_thought = torch.relu(raw_electrical_signal)
        
        return activated_thought

# --- LOCAL VERIFICATION RUN ---
if __name__ == "__main__":
    print("🧠 Testing AI Sensory Signal Transmission...")
    
    # 1. Initialize a 3-layer brain structure: 4 Inputs -> 5 Hidden States -> 2 Outputs
    architecture = [4, 5, 2]
    brain_layers = build_local_child_brain(architecture)
    
    # 2. Simulate raw sensory input (e.g., an environment state or 4 token values)
    # PyTorch tensors require float types for structural matrix operations
    mock_sensory_input = torch.tensor([[1.0, 0.5, -2.0, 3.1]])
    print(f"\n📥 Incoming Sensory Input Vector: {mock_sensory_input}")
    
    # 3. Pass the input through Layer 1
    hidden_thought = CognitiveSenses.transmit_signal(mock_sensory_input, brain_layers[0])
    print(f"🔄 Intermediate Hidden State (Layer 1 output): {hidden_thought.data}")
    
    # 4. Pass the hidden thought through Layer 2 to get the final vocal/action output
    final_output = CognitiveSenses.transmit_signal(hidden_thought, brain_layers[1])
    print(f"📤 Final Output Response (Layer 2 output): {final_output.data}")
