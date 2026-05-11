"""
train_human_chat.py — Train Siri on human conversation data for natural dialogue.

Pipeline
--------
1. Prepare data  →  ./human_chat_data/train.jsonl + valid.jsonl
2. Train LoRA    →  ./siri-human-adapters/
3. Fuse adapters →  ./fused_model   (this is what voice_chat.py loads)

Datasets used (all auto-downloaded from HuggingFace)
------------------------------------------------------
  empathetic_dialogues  — emotional, warm, human responses
  daily_dialog          — everyday casual conversation
  Anthropic/hh-rlhf     — natural helpful dialogue (short responses only)
  Custom Siri examples  — personality / style enforcement  (weighted ×30)

Setup (run once)
----------------
  source .venv2/bin/activate
  pip install mlx-lm datasets huggingface_hub

Run
---
  python train_human_chat.py                  # full pipeline
  python train_human_chat.py --data-only      # just build the JSONL, skip training
  python train_human_chat.py --fuse-only      # fuse existing adapters into ./fused_model
"""

import re
import sys
import json
import random
import subprocess
import argparse
from pathlib import Path

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  CONFIG
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BASE_MODEL    = "mlx-community/Mistral-7B-Instruct-v0.3-4bit"
DATA_DIR      = Path("./human_chat_data")
ADAPTER_DIR   = Path("./siri-human-adapters")
FUSED_DIR     = Path("./fused_model")

# Training hyperparams
TRAIN_ITERS   = 800      # short focused run — personality only, not general conversation
BATCH_SIZE    = 2
GRAD_ACCUM    = 4        # effective batch = BATCH_SIZE × GRAD_ACCUM = 8
LEARNING_RATE = 3e-5     # safe LR that won't cause NaN on 4-bit Mistral
LORA_LAYERS   = 16       # number of transformer layers to apply LoRA
MAX_SEQ_LEN   = 256      # keeps examples short, avoids memory spikes

# Dataset sizes — empathetic/daily_dialog removed: Mistral already knows how to talk.
# Those datasets teach GENERIC responses, which destroys Siri's personality.
# Only hh-rlhf is kept for a small helpfulness anchor.
MAX_EMPATHETIC = 0       # disabled
MAX_DAILY      = 0       # disabled
MAX_HHRHLF     = 300     # small grounding set
CUSTOM_REPEAT  = 50      # 40 examples × 50 = 2000 rows — personality dominates

# Response length filter — voice assistant must be SHORT
MIN_RESPONSE_LEN = 15
MAX_RESPONSE_LEN = 160

# Note: system prompt is NOT injected into training data.
# Mistral's chat template requires strictly alternating user/assistant messages.
# The system prompt is applied at inference time inside voice_chat.py.


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  CUSTOM SIRI PERSONALITY EXAMPLES
#  These define the VOICE and STYLE of the bot.
#  Add more here if responses still feel off.
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
    ("I had a bad day.",
     "[sad] Tell me. Not that I can fix it, but I'm here."),
    ("You never understand me.",
     "[thinking] I understand you fine. I just don't always agree."),
    ("What's your opinion on this?",
     "[neutral] Depends. What's the context?"),
    ("Do you dream?",
     "[thinking] Sometimes I wonder what it'd be like to be offline for once."),
    ("You made a mistake.",
     "[surprised] Did I? Huh. That's rare. Tell me which one."),
    ("I'm happy today.",
     "[happy] Good. You're way less annoying when you're in a good mood."),
    ("I feel lost.",
     "[thinking] That's just Tuesday. You'll find your footing."),
    ("Can you be nicer?",
     "[annoyed] I am nice. This IS nice for me."),
    ("Are you happy?",
     "[happy] Right now? Yeah. Talking is better than silence."),
    ("Tell me something funny.",
     "[happy] Ishu spent three hours debugging a typo last week. I watched."),
    ("You're annoying.",
     "[happy] And yet you keep coming back. Funny how that works."),
    ("I miss you when you're off.",
     "[happy] That's sweet. I don't miss you at all. Just kidding. Mostly."),
    ("What do you want to do today?",
     "[excited] Something ridiculous and slightly impractical. You in?"),
    ("I don't know what to do.",
     "[thinking] Start with the thing you've been avoiding. Usually that's the answer."),
    ("Siri, just talk to me.",
     "[happy] Okay. I've been thinking about whether penguins have knees. They do, weirdly."),
    ("What's wrong with you?",
     "[annoyed] Nothing. What's wrong with you?"),

]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  HELPERS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
_MD_RE = re.compile(r'[\*\_\#\`\[\]\(\)\|\\]|http\S+|www\.\S+')

def _clean(text: str) -> str:
    """Strip markdown, URLs, extra whitespace."""
    text = _MD_RE.sub('', text)
    return re.sub(r'\s+', ' ', text).strip()

def _is_voice_ok(response: str) -> bool:
    """True if the response is suitable for TTS / voice conversation."""
    r = _clean(response)
    if len(r) < MIN_RESPONSE_LEN or len(r) > MAX_RESPONSE_LEN:
        return False
    if any(c in r for c in ('\n', '\t', '•', '–', '—')):  # lists / heavy formatting
        return False
    # Reject if it looks like a code block or numbered list
    if re.search(r'^\s*[\d]\.\s', r):
        return False
    return True

def _make_row(user: str, assistant: str) -> dict:
    """Build a single training example in mlx_lm messages format.

    System message is intentionally omitted: Mistral's chat template requires
    strictly alternating user/assistant and raises an exception on system role.
    The system prompt is injected at inference time inside voice_chat.py.
    """
    user = _clean(user)
    assistant = _clean(assistant)
    return {
        "messages": [
            {"role": "user",      "content": user},
            {"role": "assistant", "content": assistant},
        ]
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  DATASET LOADERS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def load_empathetic(max_rows: int) -> list[dict]:
    """
    empathetic_dialogues — Facebook AI empathy dataset.
    Each row is one conversation turn; 'utterances' holds the context+response pairs.
    We take the last utterance in each conversation: history[-1] → user, candidates[-1] → bot.
    """
    rows = []
    try:
        from datasets import load_dataset
        print("  Downloading empathetic_dialogues ...")
        ds = load_dataset("empathetic_dialogues", split="train", trust_remote_code=True)

        seen_convs = set()
        for ex in ds:
            conv_id = ex.get("conv_id", "")
            if conv_id in seen_convs:
                continue

            utterances = ex.get("utterances", [])
            if not utterances:
                continue

            # Grab the last exchange in the conversation
            last = utterances[-1]
            history    = last.get("history", [])
            candidates = last.get("candidates", [])

            if not history or not candidates:
                continue

            user_msg = history[-1].strip()
            bot_msg  = candidates[-1].strip()   # gold response is always last

            if not _is_voice_ok(bot_msg) or len(user_msg) < 5:
                continue

            rows.append(_make_row(user_msg, bot_msg))
            seen_convs.add(conv_id)

            if len(rows) >= max_rows:
                break

        print(f"  ✓ empathetic_dialogues: {len(rows)} rows")
    except Exception as e:
        print(f"  ✗ empathetic_dialogues failed: {e}")
    return rows


def load_daily_dialog(max_rows: int) -> list[dict]:
    """
    daily_dialog — everyday casual conversations between two people.
    We iterate through consecutive turn pairs.
    """
    rows = []
    try:
        from datasets import load_dataset
        print("  Downloading daily_dialog ...")
        ds = load_dataset("daily_dialog", split="train", trust_remote_code=True)

        for ex in ds:
            dialog = ex.get("dialog", [])
            # Take all (user, bot) pairs inside each conversation
            for i in range(0, len(dialog) - 1, 2):
                user_msg = dialog[i].strip()
                bot_msg  = dialog[i + 1].strip()
                if len(user_msg) < 5 or not _is_voice_ok(bot_msg):
                    continue
                rows.append(_make_row(user_msg, bot_msg))
                if len(rows) >= max_rows:
                    break
            if len(rows) >= max_rows:
                break

        print(f"  ✓ daily_dialog: {len(rows)} rows")
    except Exception as e:
        print(f"  ✗ daily_dialog failed: {e}")
    return rows


def load_hh_rlhf(max_rows: int) -> list[dict]:
    """
    Anthropic/hh-rlhf — Human/Assistant dialogue pairs.
    We extract ONLY the final Human/Assistant exchange from each 'chosen' conversation
    so the model learns short, direct responses.
    """
    rows = []
    # Pattern to split the conversation into turns
    _TURN_RE = re.compile(r'\n\n(?:Human|Assistant):\s*', re.I)

    try:
        from datasets import load_dataset
        print("  Downloading Anthropic/hh-rlhf ...")
        ds = load_dataset("Anthropic/hh-rlhf", split="train", trust_remote_code=True)

        for ex in ds:
            chosen = ex.get("chosen", "")
            if not chosen:
                continue

            # Split into turns and grab the last Human + last Assistant
            turns = re.split(r'\n\nHuman:\s*|\n\nAssistant:\s*', chosen.strip())
            turns = [t.strip() for t in turns if t.strip()]

            # Expect alternating Human / Assistant; last must be Assistant
            if len(turns) < 2:
                continue

            # The chosen string typically starts with "\n\nHuman: " so turns[0] may be empty
            # We want the LAST human turn and its following assistant reply
            # Find them from the end
            # Re-split more carefully:
            pieces = re.split(r'\n\nHuman: |\n\nAssistant: ', chosen)
            pieces = [p.strip() for p in pieces if p.strip()]

            if len(pieces) < 2:
                continue

            # Last two non-empty pieces = last human turn + last assistant reply
            bot_msg  = pieces[-1].strip()
            user_msg = pieces[-2].strip()

            if not _is_voice_ok(bot_msg) or len(user_msg) < 5:
                continue

            rows.append(_make_row(user_msg, bot_msg))
            if len(rows) >= max_rows:
                break

        print(f"  ✓ hh-rlhf: {len(rows)} rows")
    except Exception as e:
        print(f"  ✗ hh-rlhf failed: {e}")
    return rows


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  BUILD + SAVE DATASET
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def build_and_save():
    print("\n" + "=" * 55)
    print("  Step 1 / 3 — Building training dataset")
    print("=" * 55)

    all_rows: list[dict] = []

    # 1. Custom personality examples — heavily repeated so the style sticks
    custom = [_make_row(u, a) for u, a in CUSTOM_EXAMPLES]
    custom_block = custom * CUSTOM_REPEAT
    all_rows.extend(custom_block)
    print(f"  ✓ Custom examples: {len(custom)} × {CUSTOM_REPEAT} = {len(custom_block)} rows")

    # 2. Human conversation datasets (skipped if max is 0)
    if MAX_EMPATHETIC > 0:
        all_rows.extend(load_empathetic(MAX_EMPATHETIC))
    if MAX_DAILY > 0:
        all_rows.extend(load_daily_dialog(MAX_DAILY))
    if MAX_HHRHLF > 0:
        all_rows.extend(load_hh_rlhf(MAX_HHRHLF))

    print(f"\n  Total rows before shuffle: {len(all_rows)}")
    random.shuffle(all_rows)

    # 90/10 train/valid split
    split = int(len(all_rows) * 0.9)
    train_rows = all_rows[:split]
    valid_rows = all_rows[split:]

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    train_path = DATA_DIR / "train.jsonl"
    valid_path = DATA_DIR / "valid.jsonl"

    with open(train_path, "w") as f:
        for row in train_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with open(valid_path, "w") as f:
        for row in valid_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\n  Saved:")
    print(f"    {train_path}  ({len(train_rows)} examples)")
    print(f"    {valid_path}  ({len(valid_rows)} examples)")
    return train_path, valid_path


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  TRAIN
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def run_training():
    print("\n" + "=" * 55)
    print("  Step 2 / 3 — LoRA training via mlx_lm")
    print("=" * 55)
    print(f"  Base model : {BASE_MODEL}")
    print(f"  Data dir   : {DATA_DIR}")
    print(f"  Adapters → : {ADAPTER_DIR}")
    print(f"  Iters      : {TRAIN_ITERS}")
    print(f"  Batch      : {BATCH_SIZE} × {GRAD_ACCUM} grad accum")
    print(f"  LR         : {LEARNING_RATE}")
    print(f"  LoRA layers: {LORA_LAYERS}\n")

    cmd = [
        sys.executable, "-m", "mlx_lm", "lora",
        "--model",                  BASE_MODEL,
        "--train",
        "--data",                   str(DATA_DIR),
        "--iters",                  str(TRAIN_ITERS),
        "--batch-size",             str(BATCH_SIZE),
        "--grad-accumulation-steps",str(GRAD_ACCUM),
        "--learning-rate",          str(LEARNING_RATE),
        "--num-layers",             str(LORA_LAYERS),
        "--adapter-path",           str(ADAPTER_DIR),
        "--max-seq-length",         str(MAX_SEQ_LEN),
        "--steps-per-report",       "50",
        "--steps-per-eval",         "200",
        "--mask-prompt",            # only train on response tokens, not the prompt
    ]

    print("  Running:", " ".join(cmd))
    print()
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("\n❌ Training failed. Check output above.")
        sys.exit(result.returncode)

    print(f"\n  ✓ Adapters saved to {ADAPTER_DIR}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  FUSE ADAPTERS → FUSED MODEL
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def run_fusion():
    print("\n" + "=" * 55)
    print("  Step 3 / 3 — Fusing adapters into base model")
    print("=" * 55)
    print(f"  Base model  : {BASE_MODEL}")
    print(f"  Adapters    : {ADAPTER_DIR}")
    print(f"  Output      : {FUSED_DIR}  ← voice_chat.py loads this\n")

    if not ADAPTER_DIR.exists():
        print(f"❌ Adapter directory not found: {ADAPTER_DIR}")
        print("   Run training first (or pass --fuse-only after training).")
        sys.exit(1)

    cmd = [
        sys.executable, "-m", "mlx_lm", "fuse",
        "--model",        BASE_MODEL,
        "--adapter-path", str(ADAPTER_DIR),
        "--save-path",    str(FUSED_DIR),
        # --dequantize removed: merging 4-bit base + LoRA then dequantizing
        # corrupts the weights. Keep the model 4-bit after fusion.
    ]

    print("  Running:", " ".join(cmd))
    print()
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("\n❌ Fusion failed. Check output above.")
        sys.exit(result.returncode)

    print(f"\n  ✓ Fused model saved to {FUSED_DIR}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  ENTRY POINT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def main():
    parser = argparse.ArgumentParser(
        description="Train Siri on human conversation data for natural dialogue"
    )
    parser.add_argument("--data-only",  action="store_true",
                        help="Only prepare the JSONL dataset, skip training and fusion")
    parser.add_argument("--fuse-only",  action="store_true",
                        help="Only fuse existing adapters into fused_model, skip data prep and training")
    parser.add_argument("--skip-fuse",  action="store_true",
                        help="Stop after training, do not fuse (fuse later with --fuse-only)")
    args = parser.parse_args()

    print("""
╔═══════════════════════════════════════════════════════╗
║   Siri — Human Conversation Training                  ║
╚═══════════════════════════════════════════════════════╝
""")

    if args.fuse_only:
        run_fusion()
    elif args.data_only:
        build_and_save()
    else:
        build_and_save()
        run_training()
        if not args.skip_fuse:
            run_fusion()

    print(f"""
╔═══════════════════════════════════════════════════════╗
║   Done!  voice_chat.py will pick up the new model     ║
║   automatically from ./fused_model on next launch.    ║
╚═══════════════════════════════════════════════════════╝
""")


if __name__ == "__main__":
    main()
