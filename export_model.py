from mlx_tune import FastLanguageModel

# Load the base model and adapters
print("Loading model and adapters...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="mlx-community/Mistral-7B-Instruct-v0.3-4bit",
    max_seq_length=1024,
    load_in_4bit=True,
)
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_alpha=16,
    lora_dropout=0.05,
)
print("Loading adapter weights...")
model.load_adapter("./siri-mistral")

print("Exporting GGUF...")
model.save_pretrained_gguf("./siri-mistral-fp16", tokenizer)
print("Done!")
