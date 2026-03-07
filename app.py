from flask import Flask, render_template, request, send_file
from nlp_engine import evaluate_answer
import pandas as pd
import os
from collections import Counter

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    results = None
    if request.method == "POST":
        teacher_ans = request.form.get("teacher_answer")
        student_ans = request.form.get("student_answer")
        total_marks = float(request.form.get("total_marks"))
        
        raw_concepts = request.form.get("concepts", "")
        concepts = [c.strip() for c in raw_concepts.split(",") if c.strip()]
        
        results = evaluate_answer(teacher_ans, student_ans, concepts, total_marks)
        
    return render_template("index.html", results=results, active_tab='single')

# --- TIER 2 & ANALYTICS: BULK CSV GRADING ---
@app.route("/bulk", methods=["POST"])
def bulk_grade():
    file = request.files.get("csv_file")
    teacher_ans = request.form.get("bulk_teacher_answer")
    total_marks = float(request.form.get("bulk_total_marks"))
    
    raw_concepts = request.form.get("bulk_concepts", "")
    concepts = [c.strip() for c in raw_concepts.split(",") if c.strip()]

    if not file:
        return "No file uploaded!", 400

    df = pd.read_csv(file)
    
    awarded_marks_list = []
    feedback_list = []
    all_missing_concepts = [] # Track missed concepts for the whole class
    
    for index, row in df.iterrows():
        student_ans = str(row.get('Student_Answer', ''))
        
        res = evaluate_answer(teacher_ans, student_ans, concepts, total_marks)
        awarded_marks_list.append(res['awarded_marks'])
        
        # Add to our global list of missing concepts for the analytics chart
        all_missing_concepts.extend(res['missing_concepts'])
        
        feedback = f"Match: {res['semantic_similarity']}%. "
        if res['missing_concepts']:
            feedback += f"Missing: {', '.join(res['missing_concepts'])}. "
        else:
            feedback += "All concepts found! "
            
        if res['irrelevant_sentences']:
            feedback += f"Fluff detected: {len(res['irrelevant_sentences'])} sentence(s)."
            
        feedback_list.append(feedback.strip())

    df['Awarded_Marks'] = awarded_marks_list
    df['AI_Feedback'] = feedback_list

    # Save to a local file in your project folder so the teacher can download it later
    output_path = "graded_results.csv"
    df.to_csv(output_path, index=False)

    # --- CALCULATE ANALYTICS FOR CHART.JS ---
    total_students = len(df)
    avg_score = round(sum(awarded_marks_list) / total_students, 1) if total_students > 0 else 0
    
    # 1. Calculate Grade Distribution (0-20%, 20-40%, etc.)
    dist = {"0-20%": 0, "21-40%": 0, "41-60%": 0, "61-80%": 0, "81-100%": 0}
    for mark in awarded_marks_list:
        p = (mark / total_marks) * 100
        if p <= 20: dist["0-20%"] += 1
        elif p <= 40: dist["21-40%"] += 1
        elif p <= 60: dist["41-60%"] += 1
        elif p <= 80: dist["61-80%"] += 1
        else: dist["81-100%"] += 1

    # 2. Calculate Most Missed Concepts
    concept_counts = Counter(all_missing_concepts)
    
    # Package everything up to send to the HTML page
    bulk_results = {
        "total_students": total_students,
        "average_score": avg_score,
        "total_marks": total_marks,
        "dist_labels": list(dist.keys()),
        "dist_data": list(dist.values()),
        "concept_labels": list(concept_counts.keys()),
        "concept_data": list(concept_counts.values())
    }

    # Tell the HTML to load the bulk tab automatically
    return render_template("index.html", bulk_results=bulk_results, active_tab='bulk')

# --- ROUTE TO DOWNLOAD THE CSV AFTER VIEWING THE DASHBOARD ---
@app.route("/download")
def download_csv():
    return send_file("graded_results.csv", as_attachment=True, download_name="Graded_Class_Results.csv")

if __name__ == "__main__":
    print("Starting Flask Web Server...")
    app.run(debug=True)