import torch
# We import our structural tissue and processing units
from brain_tissue import LocalNeuralLayer, build_local_child_brain
from senses import CognitiveSenses

class ParentalCorrection:
    """
    🗣️ Acts as the parent teaching the blank-slate AI child by evaluating errors 
    and correcting internal weight configurations.
    """
    @staticmethod
    def calculate_error(predicted_output: torch.Tensor, target_output: torch.Tensor) -> torch.Tensor:
        """📏 Calculates Mean Squared Error (how far off the child was from the target)."""
        return torch.mean((predicted_output - target_output) ** 2)

    @staticmethod
    def nudge_memory(layers: list, learning_rate: float = 0.01):
        """⚖️ Step-by-step shifts synaptic weights based on the calculated error adjustments."""
        # torch.no_grad() prevents tracking these structural updates as new errors
        with torch.no_grad():
            for layer in layers:
                if layer.weights.grad is not None:
                    # Nudge the weights in the opposite direction of the error
                    layer.weights -= learning_rate * layer.weights.grad
                    layer.biases -= learning_rate * layer.biases.grad
                    
                    # Wipe the old memory gradients clean for the next learning cycle
                    layer.weights.grad.zero_()
                    layer.biases.grad.zero_()

# --- LOCAL VERIFICATION RUN (PARENT TRAINING THE CHILD) ---
if __name__ == "__main__":
    print("👶 Initializing AI Child for its first learning session...")
    
    # 1. Wire a clean network: 3 Inputs -> 4 Hidden States -> 2 Outputs
    architecture = [3, 4, 2]
    child_brain = build_local_child_brain(architecture)
    
    # 2. Define a simple teaching lesson:
    # INPUT: Environment signals [1.0, 0.0, -0.5]
    # TARGET MANNER: The parent wants the child to output exactly [0.8, 0.1]
    lesson_input = torch.tensor([[1.0, 0.0, -0.5]])
    parent_target = torch.tensor([[0.8, 0.1]])
    
    print(f"🎯 Target Manners Goal: {parent_target.data}")
    print("\n--- Starting Training Loops ---")
    
    # 3. The Parenting Loop: We repeat the correction process 100 times
    for generation in range(101):
        # Step A: Child tries to answer using its current memory
        hidden = CognitiveSenses.transmit_signal(lesson_input, child_brain[0])
        prediction = CognitiveSenses.transmit_signal(hidden, child_brain[1])
        
        # Step B: Parent measures how bad the mistake was
        loss = ParentalCorrection.calculate_error(prediction, parent_target)
        
        # Step C: Trigger the internal calculus backpropagation reflex
        loss.backward()
        
        # Step D: Apply the correction to the weights
        ParentalCorrection.nudge_memory(child_brain, learning_rate=0.1)
        
        # Print progress updates every 20 generations
        if generation % 20 == 0:
            print(f"Generation {generation:3d} | Current Error (Loss): {loss.item():.6f} | Child Answer: {prediction.data.numpy()[0]}")
            
    print("\n🎉 Success! The Child AI successfully updated its blank memory to match the manners taught by the parent.")
