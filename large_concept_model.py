from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline
from sonar.models.blaser.loader import load_blaser_model
import torch
import json

# Load BLASER 2.0 reference-based model
model = load_blaser_model("blaser_2_0_ref").eval()

# Load the text embedding model
text_embedder = TextToEmbeddingModelPipeline(
    encoder = "text_sonar_basic_encoder",
    tokenizer = "text_sonar_basic_encoder"
)

# Your sentence
file = open('training_dataset.json')
training_data = json.loads(file.read())

# Language tags (ISO 639-3 + script)
lang = "eng_Latn"
prompt_sentence = ['cat sat on']

similarities_and_ids = {}

src_emb = text_embedder.predict(prompt_sentence, source_lang=lang)

for sentence in training_data:
 ref_emb = text_embedder.predict(sentence, source_lang=lang)

 with torch.inference_mode():
    score = model(src=src_emb, ref=ref_emb, mt=ref_emb).item()

 similarities_and_ids.update({score:training_data.index(sentence)})

print(similarities_and_ids)

predicted_sentence = training_data[similarities_and_ids.get(max(similarities_and_ids))][0].split()[len(prompt_sentence[0].split()):]

next_sentence = []

for word in predicted_sentence:
 if word not in prompt_sentence[0].split():
  next_sentence.append(word)

print(next_sentence)