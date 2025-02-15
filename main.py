import sys
import os
from scripts.extract_documents import extract_all_text, build_llama_index
from scripts.finetune_model import finetune_model
from scripts.run_model import run_model
data_folder = "data"

def main():
    print("\nSelect an option:")
    print("1. Aggregate Data from Documents (with LlamaIndex processing)")
    print("2. Fine-tune the Model (using DeepSeek-R1:1.5B)")
    print("3. Run the Fine-tuned Model (using DeepSeek-R1:1.5B)")

    choice = input("\nEnter your choice (1, 2 or 3): ").strip()

    data_folder = "data"
    model_folder = "models"  # Local folder where models are stored
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"  # Correct Hugging Face model name
    local_model_path = os.path.join(model_folder, model_name.replace("/", "_"))  # Convert model name to a local directory path

    if choice == '1':
        print("\n[INFO] Starting data aggregation and LlamaIndex processing...")

        # Extract raw text
        documents_text = extract_all_text(data_folder)
        raw_data_file = os.path.join(data_folder, "raw_data.txt")

        if documents_text:
            with open(raw_data_file, "w", encoding="utf-8") as f:
                f.write("\n".join(documents_text))
            print(f"[SUCCESS] Aggregated {len(documents_text)} documents and saved to '{raw_data_file}'.")
        else:
            print("[WARNING] No documents found in the data folder!")

        # Build LlamaIndex and export processed data
        llama_index_file = os.path.join(data_folder, "llama_index_data.txt")
        build_llama_index(data_folder, output_file=llama_index_file)

        # Merge raw data and LlamaIndex processed data
        merged_file = os.path.join(data_folder, "merged_data.txt")
        try:
            with open(merged_file, "w", encoding="utf-8") as out_f:
                if os.path.exists(raw_data_file):
                    with open(raw_data_file, "r", encoding="utf-8") as f:
                        out_f.write(f.read() + "\n")
                if os.path.exists(llama_index_file):
                    with open(llama_index_file, "r", encoding="utf-8") as f:
                        out_f.write(f.read() + "\n")

            print(f"[SUCCESS] Merged data saved to '{merged_file}'. Use this file for fine-tuning.")
        except Exception as e:
            print(f"[ERROR] Failed to merge data: {str(e)}")

    elif choice == '2':
        print("\n[INFO] Starting model fine-tuning with DeepSeek-R1:1.5B...")

        # Check if the model exists locally, otherwise download from Hugging Face
        if os.path.exists(local_model_path):
            print(f"[INFO] Using local model from: {local_model_path}")
            finetune_model(local_model_path)
        else:
            print(f"[WARNING] Local model not found. Attempting to download: {model_name}")
            finetune_model(model_name)

    elif choice == '3':
        print("\n[INFO] Running the fine-tuned model...")

        # Determine model location
        model_path = local_model_path if os.path.exists(local_model_path) else model_name

        # Get user input
        prompt = input("\nEnter the prompt for the model: ").strip()

        # Run model inference
        try:
            response = run_model(model_path, prompt)
            if response:
                print(f"\n[MODEL RESPONSE]: {response}")
            else:
                print("[ERROR] Model failed to generate a response.")
        except Exception as e:
            print(f"[ERROR] Failed to run the model: {str(e)}")

    else:
        print("[ERROR] Invalid choice. Please enter 1, 2, or 3.")
        sys.exit(1)

if __name__ == "__main__":
    main()
