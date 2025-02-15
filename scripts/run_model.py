from transformers import AutoModelForCausalLM, AutoTokenizer

def run_model(model_name="local-ollama/LLaMA2-2B", prompt=""):
    try:
        print(f"Loading the fine-tuned model: {model_name}")
        model = AutoModelForCausalLM.from_pretrained(f"./model/{model_name.replace('/', '_')}")
        tokenizer = AutoTokenizer.from_pretrained(f"./model/{model_name.replace('/', '_')}")
        
        # Tokenize the input prompt
        inputs = tokenizer(prompt, return_tensors="pt").input_ids
        
        # Generate the model output
        output = model.generate(inputs, max_length=100, do_sample=True, top_k=50)
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        
        return generated_text
    except Exception as e:
        print(f"Error running the model: {e}")
        return None
