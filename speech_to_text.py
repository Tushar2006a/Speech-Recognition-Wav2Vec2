import torch
import librosa
from transformers import Wav2Vec2Tokenizer, Wav2Vec2ForCTC

# Load pretrained model
tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

# Load audio file
audio, _ = librosa.load("audio.wav", sr=16000)

# Convert audio to input values
inputs = tokenizer(audio, return_tensors="pt", padding="longest").input_values

# Run model
with torch.no_grad():
    logits = model(inputs).logits

# Decode output
predicted_ids = torch.argmax(logits, dim=-1)
text = tokenizer.decode(predicted_ids[0])

print("\nTranscribed Text:")
print(text.lower())
