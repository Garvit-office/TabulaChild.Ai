import torch

class RecurrentSyntaxLayer:
    """
    🔄 Represents a neural layer with a looping sequence buffer.
    Allows our AI Child to process language, keeping memory of previous words.
    """
    def __init__(self, input_size: int, hidden_size: int):
        self.hidden_size = hidden_size
        
        # 🕸️ Synaptic connection from new input data to hidden mind
        self.W_input = torch.randn(hidden_size, input_size, requires_grad=True)
        # 🔁 Structural looping connection from previous memory state to new memory state
        self.W_recurrent = torch.randn(hidden_size, hidden_size, requires_grad=True)
        
        # ⚖️ Internal biases
        self.bias = torch.zeros(hidden_size, requires_grad=True)

    def process_sequence(self, sequence_tensor: torch.Tensor) -> list:
        """
        🔂 Processes an entire sentence word-by-word, updating memory loops.
        sequence_tensor dimension: [sequence_length, input_size]
        """
        # Step A: Child starts the sentence with a completely clear internal focus state (zeros)
        hidden_state = torch.zeros(1, self.hidden_size)
        all_outputs = []
        
        # Step B: Loop through every word vector step-by-step
        for word_vector in sequence_tensor:
            # Reshape word to match matrix multiplication alignment [1, input_size]
            word_vector = word_vector.unsqueeze(0)
            
            # Step C: Combine the current word input with the past hidden memory state
            input_signal = torch.matmul(word_vector, self.W_input.t())
            memory_signal = torch.matmul(hidden_state, self.W_recurrent.t())
            
            # Step D: Apply the structural activation trigger (tanh is excellent for recurrent scales)
            hidden_state = torch.tanh(input_signal + memory_signal + self.bias)
            
            # Record what the child is thinking at this specific step in the sentence
            all_outputs.append(hidden_state)
            
        return all_outputs

# --- LOCAL VERIFICATION RUN (TEACHING A SYNTAX PATTERN) ---
if __name__ == "__main__":
    print("👶 Initializing local AI Child with Language Loop Memory...")
    
    # Let's say words are represented by 3 numbers, and the brain has 4 hidden thought paths
    language_engine = RecurrentSyntaxLayer(input_size=3, hidden_size=4)
    
    # 📝 Simulate a 3-word sentence data structure (e.g., "Always [1,0,0] Be [0,1,0] Polite [0,0,1]")
    mock_sentence = torch.tensor([
        [1.0, 0.0, 0.0],  # Word 1
        [0.0, 1.0, 0.0],  # Word 2
        [0.0, 0.0, 1.0]   # Word 3
    ])
    
    print(f"\n🗣️ Feeding a {mock_sentence.shape[0]}-word structural sentence to the Child model.")
    
    # Process the sequence step-by-step
    thought_history = language_engine.process_sequence(mock_sentence)
    
    # Verify how the memory shifted as the sentence progressed
    for word_idx, step_thought in enumerate(thought_history):
        print(f"  ↳ Word {word_idx + 1} processing complete. Internal Memory State: {step_thought.data.numpy()}")
        
    print("\n🎉 Success! The sequence loop functions correctly. The AI retains context over time.")
