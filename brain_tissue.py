import torch

class LocalNeuralLayer:
    """
    🧠 Represents a blank-slate neural layer powered by local PyTorch matrices.
    No cloud APIs, no external keys. Completely isolated on your machine.
    """
    def __init__(self, input_size: int, output_size: int):
        # 🕸️ The Synaptic Memory Matrix
        # We initialize random weights using a normal distribution (mean=0, std=1)
        # requires_grad=True tells the local engine to prepare this memory for learning calculus later
        self.weights = torch.randn(output_size, input_size, requires_grad=True)
        
        # ⚖️ The Bias Vector initialized to structural zeros
        self.biases = torch.zeros(output_size, requires_grad=True)

    def display_dimensions(self):
        """📊 Inspects the exact biological wiring of this local layer."""
        print(f"Layer Memory Matrix -> Rows (Neurons): {self.weights.shape[0]}, Columns (Inputs): {self.weights.shape[1]}")

def build_local_child_brain(layer_sizes: list):
    """🏗️ Chains local layers together sequentially to form the blank mind."""
    network = []
    for i in range(len(layer_sizes) - 1):
        layer = LocalNeuralLayer(input_size=layer_sizes[i], output_size=layer_sizes[i+1])
        network.append(layer)
    return network

# --- LOCAL VERIFICATION RUN ---
if __name__ == "__main__":
    print("👶 Initializing local 'Tabula Rasa' Child Model...")
    
    # Defining an architecture: 3 Inputs -> 5 Hidden Behaviors -> 2 Final Outputs
    architecture = [3, 5, 2]
    child_brain = build_local_child_brain(architecture)
    
    # Print out the local architecture shapes to verify accuracy
    for index, layer in enumerate(child_brain):
        print(f"\n[Layer {index + 1} Wiring]")
        layer.display_dimensions()
        print(f"Raw Weights Tensor:\n{layer.weights.data}")
