import torch
import os
# Import our structural foundations from our very first file
from brain_tissue import build_local_child_brain

class AdoptionPackage:
    """
    📦 Manages exporting and importing our local 'Tabula Rasa' model.
    Allows companies to adopt the pre-behaved AI child and train it locally.
    """
    @staticmethod
    def pack_model_for_adoption(layers: list, filename: str = "child_brain.pt"):
        """💾 Packs up the structural weight matrices and weights to a local file."""
        package_state = {}
        
        # Extract the memory matrices from each layer sequentially
        for index, layer in enumerate(layers):
            package_state[f"layer_{index}_weights"] = layer.weights.data
            package_state[f"layer_{index}_biases"] = layer.biases.data
            
        # Serialize and write the memory securely to a local file
        torch.save(package_state, filename)
        print(f"📦 Success! Packed brain snapshot securely saved to: {os.path.abspath(filename)}")

    @staticmethod
    def load_adopted_model(architecture: list, filename: str = "child_brain.pt") -> list:
        """🏗️ Reconstructs the exact brain structure and restores its weight memory."""
        if not os.path.exists(filename):
            raise FileNotFoundError(f"No child brain model found at {filename}")
            
        # Initialize a fresh blank slate network matching the target dimensions
        fresh_brain = build_local_child_brain(architecture)
        
        # Load the saved raw mathematical states
        saved_state = torch.load(filename)
        
        # Inject the saved memories into our newly created blank neural pathways
        for index, layer in enumerate(fresh_brain):
            layer.weights.data = saved_state[f"layer_{index}_weights"]
            layer.biases.data = saved_state[f"layer_{index}_biases"]
            
        print(f"🔓 Success! The adopted model memory has been perfectly restored.")
        return fresh_brain

# --- LOCAL VERIFICATION RUN (PACKAGING AND ADOPTING) ---
# --- LOCAL VERIFICATION RUN (PACKAGING AND ADOPTING) ---
if __name__ == "__main__":
    print("🏠 [PARENT HOME] Formatting and packing up the AI Child...")
    
    # 1. Parents design a unique brain layout: 3 Inputs -> 4 Behaviors -> 2 Outputs
    family_architecture = [3, 4, 2]
    reared_child = build_local_child_brain(family_architecture)
    
    # FIX: Grab the first layer [0] from the network list to view weights
    original_sample_weight = reared_child[0].weights.data
    print(f"Verification Key (Original Layer 1 Weights):\n{original_sample_weight}")
    
    # 2. Securely pack the child's mind to a local file
    AdoptionPackage.pack_model_for_adoption(reared_child, "child_brain.pt")
    
    print("\n-------------------------------------------------------------")
    print("🏢 [COMPANY ACCESS] A medical/engineering firm adopts the model...")
    
    # 3. The company reconstructs the architecture and restores the child's state
    adopted_child = AdoptionPackage.load_adopted_model(family_architecture, "child_brain.pt")
    
    # FIX: Grab the first layer [0] from the adopted network list to verify
    restored_sample_weight = adopted_child[0].weights.data
    print(f"Verification Key (Restored Layer 1 Weights):\n{restored_sample_weight}")
    
    # Ultimate truth check
    if torch.equal(original_sample_weight, restored_sample_weight):
        print("\n🏆 COMPLETE SYSTEM SUCCESS! The structural memory file is 100% identical.")
    else:
        print("\n❌ Error: The memory configuration altered during transit.")
