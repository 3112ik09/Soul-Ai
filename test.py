"""
test_vad.py — Test mic + Silero VAD
Run: python test_vad.py
Speak and watch the bars. Should go high when you talk.
"""

import torch
import sounddevice as sd
import numpy as np

SAMPLE_RATE = 16000
CHUNK_MS    = 32
THRESHOLD   = 0.3

print("Loading Silero VAD...")
model, _ = torch.hub.load('snakers4/silero-vad', 'silero_vad', trust_repo=True)
print("✅ VAD loaded\n")

print("Available audio devices:")
print(sd.query_devices())
print(f"\nDefault input device: {sd.default.device}")
print("\n" + "="*50)
print("Speak into your mic — watch the bars")
print("Ctrl+C to stop")
print("="*50 + "\n")

chunk_size = 512  # Silero requires exactly 512 samples at 16000Hz

try:
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                        dtype='float32', blocksize=chunk_size,
                        device=1) as stream:
        while True:
            chunk, overflowed = stream.read(chunk_size)
            chunk = chunk.flatten()

            # Raw volume
            volume = float(np.abs(chunk).mean())

            # VAD probability
            tensor = torch.from_numpy(chunk)
            try:
                prob = model(tensor, SAMPLE_RATE).item()
            except Exception as e:
                prob = 0.0
                print(f"VAD error: {e}")

            # Display
            vol_bar = '█' * int(volume * 200)
            vad_bar = '█' * int(prob * 30)
            status  = '🎤 SPEAKING' if prob > THRESHOLD else '   silent '

            print(f"Vol:{volume:.4f} {vol_bar[:20]:<20} | VAD:{prob:.2f} {vad_bar[:30]:<30} | {status}",
                  end='\r', flush=True)

except KeyboardInterrupt:
    print("\n\nDone.")