import spacy
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# 1. Load the AI Models (We do this once so it's fast)
print("Loading AI Models...")
nlp = spacy.load("en_core_web_sm")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("Models loaded successfully!\n")

def evaluate_answer(teacher_answer, student_answer, key_concepts, total_marks):
    # --- STEP 1: SEMANTIC SIMILARITY (Meaning match) ---
    # Convert both answers into numerical embeddings
    embeddings = model.encode([teacher_answer, student_answer])
    
    # Calculate how similar the two arrays of numbers are (0.0 to 1.0)
    # [0] is teacher, [1] is student
    similarity_matrix = cosine_similarity([embeddings[0]], [embeddings[1]])
    semantic_score = similarity_matrix[0][0] # Extract the raw score
    
    # --- STEP 2: CONCEPT DETECTION (Keyword match) ---
    # Convert the student's answer to lowercase to make checking easier
    student_doc = nlp(student_answer.lower())
    student_text = student_doc.text
    
    concepts_found = []
    for concept in key_concepts:
        # Check if the concept (lowercased) is somewhere in the student's text
        if concept.lower() in student_text:
            concepts_found.append(concept)
            
    # Calculate concept score (0.0 to 1.0)
    if len(key_concepts) > 0:
        concept_score = len(concepts_found) / len(key_concepts)
    else:
        concept_score = 1.0 # If teacher didn't give concepts, don't penalize
        
    # --- STEP 3: FINAL CALCULATION ---
    # Let's say Semantic Meaning is worth 60% and Keywords are worth 40%
    final_score_percentage = (semantic_score * 0.6) + (concept_score * 0.4)
    
    # Convert percentage to actual marks
    awarded_marks = round(final_score_percentage * total_marks, 1)
    
    # Ensure marks don't go below 0 or above total_marks
    awarded_marks = max(0, min(awarded_marks, total_marks))
    
    return {
        "semantic_similarity": round(semantic_score * 100, 1),
        "concepts_found": concepts_found,
        "concept_score": round(concept_score * 100, 1),
        "awarded_marks": awarded_marks,
        "total_marks": total_marks
    }

# --- TEST THE ENGINE ---
if __name__ == "__main__":
    # Simulate Teacher Input
    teacher_ans = "Plants synthesize carbohydrates during photosynthesis using sunlight, water, and carbon dioxide."
    concepts = ["photosynthesis", "sunlight", "carbon dioxide", "water"]
    marks = 5

    # Simulate Student Input
    student_ans = "Plants make their own food molecules using the sun, water, and carbon dioxide through a process called photosynthesis."

    print("Evaluating Student Answer...")
    results = evaluate_answer(teacher_ans, student_ans, concepts, marks)

    print("\n--- RESULTS ---")
    print(f"Semantic Match: {results['semantic_similarity']}%")
    print(f"Concepts Found: {results['concepts_found']} ({results['concept_score']}%)")
    print(f"Final Marks Awarded: {results['awarded_marks']} / {results['total_marks']}")