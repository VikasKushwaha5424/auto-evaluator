import spacy
from sentence_transformers import SentenceTransformer

print("Loading spaCy model...")
nlp = spacy.load("en_core_web_sm")
doc = nlp("Plants synthesize carbohydrates during photosynthesis.")
print("spaCy loaded successfully! Tokens found:")
for token in doc:
    print(f" - {token.text}")

print("\nLoading Sentence-BERT model (this might take a minute as it downloads the model files)...")
model = SentenceTransformer('all-MiniLM-L6-v2')
sentences = ["This is a model answer.", "This is a student answer."]
embeddings = model.encode(sentences)
print("Sentence-BERT loaded successfully!")
print(f"Created embeddings of shape: {embeddings.shape}")