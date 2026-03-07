from flask import Flask, render_template, request
from nlp_engine import evaluate_answer # Importing your AI logic!

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    results = None
    
    if request.method == "POST":
        # 1. Get the data typed into the HTML form
        teacher_ans = request.form.get("teacher_answer")
        student_ans = request.form.get("student_answer")
        total_marks = float(request.form.get("total_marks"))
        
        # Split the comma-separated concepts into a list, remove empty spaces
        raw_concepts = request.form.get("concepts", "")
        concepts = [c.strip() for c in raw_concepts.split(",") if c.strip()]
        
        # 2. Run the data through your AI engine
        results = evaluate_answer(teacher_ans, student_ans, concepts, total_marks)
        
    # 3. Send the webpage back to the user, passing the results if they exist
    return render_template("index.html", results=results)

if __name__ == "__main__":
    # Start the web server on your local machine
    print("Starting Flask Web Server...")
    app.run(debug=True)