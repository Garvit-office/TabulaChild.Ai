import torch
import sys
from brain_tissue import build_local_child_brain
from senses import CognitiveSenses
from correction import ParentalCorrection

class LocalChatbotSystem:
    def __init__(self):
        # 1. Map simple local English vocabulary words to numeric slots
        self.vocab = {
            "<PAD>": 0, "hello": 1, "hi": 2, "how": 3, "are": 4, "you": 5,
            "i": 6, "am": 7, "happy": 8, "sad": 9, "good": 10, "bad": 11,
            "what": 12, "is": 13, "your": 14, "name": 15, "child": 16, "ai": 17
        }
        # Reverse map to convert brain math outputs back to spoken words
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        self.vocab_size = len(self.vocab)
        
        # 2. Build the Child Brain: Vocab Size Inputs -> 8 Hidden Traits -> Vocab Size Outputs
        print("👶 Initializing empty-memory interactive Chatbot...")
        self.architecture = [self.vocab_size, 8, self.vocab_size]
        self.brain = build_local_child_brain(self.architecture)
        
    def text_to_vector(self, text: str) -> torch.Tensor:
        """Converts an English word into a mathematical one-hot brain signal matrix."""
        word = text.strip().lower()
        vector = torch.zeros(1, self.vocab_size)
        if word in self.vocab:
            vector[0][self.vocab[word]] = 1.0
        else:
            # Default to greeting if word is unknown to the tiny vocab dictionary
            vector[0][self.vocab["hello"]] = 1.0
        return vector

    def vector_to_text(self, vector: torch.Tensor) -> str:
        """Converts raw neural firing outputs back into an English word."""
        predicted_index = torch.argmax(vector).item()
        return self.inverse_vocab.get(predicted_index, "unknown")

    def train_step(self, input_word: str, target_word: str):
        """LEARNING MODE: Actively updates weights to link input_word to target_word."""
        x = self.text_to_vector(input_word)
        y_target = self.text_to_vector(target_word)
        
        # Adjust synaptic paths up to 50 times quickly to commit it to memory
        for _ in range(50):
            hidden = CognitiveSenses.transmit_signal(x, self.brain)
            prediction = CognitiveSenses.transmit_signal(hidden, self.brain)
            
            loss = ParentalCorrection.calculate_error(prediction, y_target)
            loss.backward()
            ParentalCorrection.nudge_memory(self.brain, learning_rate=0.2)

    def answer_step(self, input_word: str) -> str:
        """ANSWERING MODE: Reads current weights to retrieve a text response."""
        x = self.text_to_vector(input_word)
        with torch.no_grad():
            hidden = CognitiveSenses.transmit_signal(x, self.brain)
            prediction = CognitiveSenses.transmit_signal(hidden, self.brain)
        return self.vector_to_text(prediction)

# --- RUN INTERACTIVE INTERFACE ---
if __name__ == "__main__":
    bot = LocalChatbotSystem()
    current_mode = "ANSWER" # Starts in checking mode
    
    print("\n==============================================")
    print("🤖 TABULA RASA INTERACTIVE CHATBOT RUNNING 🤖")
    print("Commands:")
    print("  /learn  -> Switch to Learning Mode (Input memory blocks)")
    print("  /answer -> Switch to Answering Mode (Test memory paths)")
    print("  /exit   -> Close application")
    print("==============================================")
    
    while True:
        try:
            user_input = input(f"\n[{current_mode} MODE] You: ").strip()
            
            if not user_input:
                continue
            if user_input.lower() == "/exit":
                print("👋 Shutting down chatbot.")
                sys.exit()
            elif user_input.lower() == "/learn":
                current_mode = "LEARN"
                print("🧠 Mode Switched: The Child AI is listening. Format your training as: word->reply")
                continue
            elif user_input.lower() == "/answer":
                current_mode = "ANSWER"
                print("💬 Mode Switched: Ask a single word to test the AI's memory matrix.")
                continue

            if current_mode == "LEARN":
                if "->" in user_input:
                    input_part, target_part = user_input.split("->", 1)
                    bot.train_step(input_part.strip(), target_part.strip())
                    print(f"✅ Memory Synapse Mapped! Linked '{input_part.strip()}' directly to '{target_part.strip()}'.")
                else:
                    print("⚠️ Invalid format! Use: input_word->desired_output_word (e.g., hello->happy)")
                    
            elif current_mode == "ANSWER":
                reply = bot.answer_step(user_input)
                print(f"🤖 AI Child: {reply}")
                
        except KeyboardInterrupt:
            print("\n👋 Shutting down safely.")
            sys.exit()
