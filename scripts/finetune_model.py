from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import Dataset
import os
import torch
data_folder = "data"

def get_model_max_length(model):
    """Retrieve the max_length from the model's config or set a default."""
    return getattr(model.config, "max_position_embeddings", 2048)

def tokenize_function(examples, tokenizer, max_length):
    """Tokenize text inputs with padding and truncation."""
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=max_length)

def finetune_model(model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"):
    # Define local model directory
    model_dir = os.path.join("model", model_name.replace("/", "_"))

    # Load model and tokenizer
    if os.path.exists(model_dir):
        print(f"Loading model from {model_dir}...")
        model = AutoModelForCausalLM.from_pretrained(model_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
    else:
        print(f"Downloading and loading model: {model_name}")
        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model.save_pretrained(model_dir)
        tokenizer.save_pretrained(model_dir)

    # Ensure the tokenizer has a pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    max_length = get_model_max_length(model)
    print(f"Using max_length={max_length} for tokenization.")

    # Load the training data
    data_file = os.path.join(data_folder, "llama_index_data.txt") # Updated filename reference
    if not os.path.exists(data_file):
        print(f"Error: {data_file} not found. Please provide the training data.")
        return

    with open(data_file, "r", encoding="utf-8") as f:
        documents_text = f.readlines()

    # Create a dataset
    data_dict = {'text': [doc.strip() for doc in documents_text if doc.strip()]}
    dataset = Dataset.from_dict(data_dict)

    # Tokenize the dataset
    dataset = dataset.map(lambda examples: tokenize_function(examples, tokenizer, max_length), batched=True)

    # Split dataset into train and test sets
    dataset = dataset.train_test_split(test_size=0.1)
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

    # Set up data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Set batch size dynamically based on GPU availability
    per_device_batch_size = 2 if torch.cuda.is_available() else 1

    # Enable bf16 if supported
    use_bf16 = torch.cuda.is_bf16_supported()

    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./finetuned_model",
        per_device_train_batch_size=per_device_batch_size,
        per_device_eval_batch_size=per_device_batch_size,
        gradient_accumulation_steps=2,
        num_train_epochs=3,  # Adjust as needed
        save_steps=500,
        save_total_limit=2,
        logging_steps=50,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        remove_unused_columns=False,
        bf16=use_bf16,  # Use bf16 only if the system supports it
        optim="adamw_torch",
        report_to="none"
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    print("Starting fine-tuning...")
    trainer.train()
    print("Fine-tuning completed.")

    # Save the final model
    final_model_path = os.path.join("finetuned_model", "final_model")
    model.save_pretrained(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    print(f"Fine-tuned model saved to {final_model_path}")

if __name__ == "__main__":
    finetune_model()
