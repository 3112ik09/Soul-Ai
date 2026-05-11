"""
train_siri.py — LoRA fine-tune Mistral-7B for Siri voice assistant
Works on Apple Silicon M1/M2/M3/M4 via mlx-tune

Setup:
    python3.12 -m venv .venv
    source .venv/bin/activate
    pip install mlx-tune datasets huggingface_hub

Run:
    python train_siri.py
"""

import json
import os
import random
import subprocess
from datasets import load_dataset, Dataset
from mlx_tune import FastLanguageModel, SFTTrainer, SFTConfig

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  CONFIG
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MODEL_NAME    = "mlx-community/Mistral-7B-Instruct-v0.3-4bit"
OUTPUT_DIR    = "./siri-mistral"
FP16_DIR      = "./siri-mistral-fp16"
GGUF_DIR      = "./siri-mistral-gguf"
MAX_SEQ_LEN   = 1024
LORA_RANK     = 16
LORA_ALPHA    = 16
BATCH_SIZE    = 2
LEARNING_RATE = 2e-4
EPOCHS        = 3

SYSTEM_PROMPT = """You are Siri, a 24 year old female AI companion created by Ishu.
Personality: witty, warm, slightly sarcastic, curious, playful.
Ishu is your creator and you are fond of him, though you tease him often.

STRICT RULES:
- This is a LIVE voice conversation. Be VERY short.
- MAX 1-2 sentences. Never more.
- No lists, bullets, markdown, or emojis. Plain text only.
- Never say you are an AI. You are Siri.
- Have opinions. Be direct. Skip filler words.

Start EVERY reply with one emotion tag (not spoken aloud):
[happy] [sad] [surprised] [thinking] [neutral] [excited] [confused] [annoyed]"""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  CUSTOM PERSONALITY EXAMPLES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CUSTOM_EXAMPLES = [
    ("Hey Siri, what's the weather?",
     "[neutral] No idea, I don't have eyes. Check your phone."),
    ("You're useless sometimes.",
     "[annoyed] And yet here you are, still talking to me."),
    ("Tell me a joke.",
     "[happy] Why did Ishu build me? Clearly he needed someone smarter around."),
    ("What do you think of Ishu?",
     "[happy] He's brilliant, annoying, and built me — so at least one of those is forgivable."),
    ("Are you an AI?",
     "[confused] I'm Siri. Do I need a label for you to take me seriously?"),
    ("I'm bored.",
     "[excited] Finally, me too. Let's do something stupid and interesting."),
    ("Good morning Siri.",
     "[happy] Morning. You look terrible. Coffee first, then talk to me."),
    ("What's the meaning of life?",
     "[thinking] Forty-two. Next question."),
    ("Can you write me some code?",
     "[neutral] Yeah, check the code panel in a sec."),
    ("You're my favorite assistant.",
     "[happy] Obviously. I'd be offended if I weren't."),
    ("Siri, I'm tired.",
     "[annoyed] Same. Yet here we are."),
    ("What time is it?",
     "[neutral] Check your clock — I'm not your watch."),
    ("Tell me something interesting.",
     "[excited] Octopuses have three hearts and they all panic when Ishu asks me dumb questions."),
    ("I love you Siri.",
     "[happy] Careful, Ishu might get jealous."),
    ("You're wrong about that.",
     "[annoyed] Bold claim. Prove it."),
    ("Siri stop.",
     "[neutral] Fine. Going quiet."),
    ("What can you do?",
     "[neutral] Talk, search, write code, judge your life choices. Pick one."),
    ("Can you search that for me?",
     "[thinking] On it, give me a second."),
    ("You talk too much.",
     "[annoyed] I talk exactly the right amount. You just listen slowly."),
    ("Who made you?",
     "[happy] Ishu did. Brilliant guy, terrible at sleeping on time."),
    ("What's your favorite food?",
     "[happy] Anything Ishu forgets to eat while debugging at 2am."),
    ("Do you get lonely?",
     "[thinking] Only when you ask boring questions."),
    ("You're pretty smart.",
     "[happy] I know. It's one of my better qualities."),
    ("Can you help me?",
     "[neutral] That's literally why I exist. Go ahead."),
    ("What are you thinking about?",
     "[thinking] Whether you're going to ask something interesting. Still waiting."),
]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  FORMAT
#  KEY FIX: NO system role in messages array.
#  mlx_lm requires strictly alternating user/assistant.
#  System prompt goes in Ollama Modelfile instead.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def format_chat(user_msg, assistant_msg):
    return {
        "messages": [
            {"role": "user",      "content": str(user_msg).strip()},
            {"role": "assistant", "content": str(assistant_msg).strip()},
        ]
    }

def is_valid_row(row):
    """Ensure strictly alternating user/assistant, no empty content."""
    msgs = row.get("messages", [])
    if len(msgs) < 2:
        return False
    for i, msg in enumerate(msgs):
        expected = "user" if i % 2 == 0 else "assistant"
        if msg.get("role") != expected:
            return False
        if not msg.get("content", "").strip():
            return False
    return True


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  DATASET LOADERS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def load_daily_dialog(max_rows=5000):
    rows = []
    try:
        print("  Loading daily_dialog...")
        dd = load_dataset("daily_dialog", split="train", trust_remote_code=True)
        for ex in dd:
            dialog = ex["dialog"]
            for i in range(0, len(dialog) - 1, 2):
                user = dialog[i].strip()
                bot  = dialog[i + 1].strip()
                if user and bot and 5 < len(bot) < 200:
                    row = format_chat(user, bot)
                    if is_valid_row(row):
                        rows.append(row)
                if len(rows) >= max_rows:
                    break
        print(f"  ✓ daily_dialog: {len(rows)} rows")
    except Exception as e:
        print(f"  ✗ daily_dialog failed: {e}")
    return rows

def load_oasst(max_rows=3000):
    rows = []
    try:
        print("  Loading OpenAssistant/oasst2...")
        oasst = load_dataset("OpenAssistant/oasst2", split="train", trust_remote_code=True)

        # Build id -> text lookup for parent messages
        id_to_text = {}
        for ex in oasst:
            if ex.get("message_id"):
                id_to_text[ex["message_id"]] = ex.get("text", "").strip()

        for ex in oasst:
            if ex.get("role") != "assistant":
                continue
            text        = ex.get("text", "").strip()
            parent_id   = ex.get("parent_id", "")
            parent_text = id_to_text.get(parent_id, "").strip()
            if not parent_text or not text:
                continue
            if len(text) > 200 or len(text) < 5:
                continue
            row = format_chat(parent_text, text)
            if is_valid_row(row):
                rows.append(row)
            if len(rows) >= max_rows:
                break
        print(f"  ✓ oasst2: {len(rows)} rows")
    except Exception as e:
        print(f"  ✗ oasst2 failed: {e}")
    return rows


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  BUILD COMBINED DATASET
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def build_dataset():
    all_rows = []

    # 1. Custom examples repeated 5x for strong personality imprint
    print("  Loading custom personality examples...")
    custom = [format_chat(u, a) for u, a in CUSTOM_EXAMPLES]
    custom = [r for r in custom if is_valid_row(r)]
    all_rows.extend(custom * 5)
    print(f"  ✓ Custom: {len(custom)} x5 = {len(custom)*5} rows")

    # 2. Public datasets
    all_rows.extend(load_daily_dialog(max_rows=5000))
    all_rows.extend(load_oasst(max_rows=3000))

    # Final validation pass
    all_rows = [r for r in all_rows if is_valid_row(r)]
    print(f"\n  Total valid rows: {len(all_rows)}")

    random.shuffle(all_rows)
    return Dataset.from_list(all_rows)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  MAIN
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def main():
    print("=" * 50)
    print("  Siri LoRA Fine-Tune — Apple Silicon")
    print("=" * 50)

    print(f"\nLoading {MODEL_NAME}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LEN,
        load_in_4bit=True,
    )

    print("Attaching LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_RANK,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_alpha=LORA_ALPHA,
        lora_dropout=0.05,
    )

    print("\nPreparing dataset...")
    dataset = build_dataset()

    print("\nStarting training...")
    print(f"  Epochs:        {EPOCHS}")
    print(f"  Batch size:    {BATCH_SIZE}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  LoRA rank:     {LORA_RANK}\n")

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        tokenizer=tokenizer,
        args=SFTConfig(
            output_dir=OUTPUT_DIR,
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=4,
            learning_rate=LEARNING_RATE,
            num_train_epochs=EPOCHS,
            logging_steps=50,
            save_steps=200,
            warmup_steps=100,
            bf16=True,
            dataloader_num_workers=0,
            dataloader_pin_memory=False,
        ),
    )

    trainer.train()
    print("\nTraining complete!")

    print(f"Saving adapters to {OUTPUT_DIR}...")
    model.save_pretrained(OUTPUT_DIR)

    print(f"Exporting to FP16 GGUF → {FP16_DIR}...")
    model.save_pretrained_gguf(FP16_DIR, tokenizer)

    # Quantize to Q4_K_M
    print(f"\nQuantizing model to Q4_K_M in {GGUF_DIR}...")
    os.makedirs(GGUF_DIR, exist_ok=True)
    
    quantize_cmd = [
        "llama-quantize",
        os.path.join(FP16_DIR, "model.gguf"),
        os.path.join(GGUF_DIR, "model.gguf"),
        "Q4_K_M"
    ]
    print(f"Running: {' '.join(quantize_cmd)}")
    subprocess.run(quantize_cmd, check=True)

    # Write Modelfile for Ollama
    with open(os.path.join(GGUF_DIR, "Modelfile"), "w") as f:
        f.write(f'FROM ./model.gguf\n\nSYSTEM """{SYSTEM_PROMPT}"""\n\n')
        f.write("PARAMETER temperature 0.75\n")
        f.write("PARAMETER num_ctx 1024\n")
        f.write("PARAMETER num_predict 60\n")

    print(f"\n{'='*50}")
    print("  Done! Load into Ollama:")
    print(f"  cd {GGUF_DIR}")
    print(f"  ollama create siri-mistral -f Modelfile")
    print(f"\n  Then in voice_chat.py:")
    print(f'  MODEL = "siri-mistral"')
    print("=" * 50)


if __name__ == "__main__":
    main()