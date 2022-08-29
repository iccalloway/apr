import torch
from transformers import Wav2Vec2Processor, HubertForCTC
from datasets import load_dataset

processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft")
model = HubertForCTC.from_pretrained("facebook/hubert-large-ls960-ft")
    
ds = load_dataset("patrickvonplaten/librispeech_asr_dummy", "clean", split="validation")

item = ds[0]['audio']['array']
print(item)
print(item.shape)
input_values = processor(item, return_tensors="pt").input_values  # Batch size 1
print(input_values)
print(input_values.shape)
logits = model(input_values).logits
predicted_ids = torch.argmax(logits, dim=-1)
transcription = processor.decode(predicted_ids[0])

print(transcription)
