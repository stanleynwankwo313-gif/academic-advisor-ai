from pathlib import Path
import pickle
from flask import Flask, render_template, request

app = Flask(__name__)
MODEL_PATH = Path(__file__).with_name('advisor_model.pkl')
if not MODEL_PATH.exists():
    raise FileNotFoundError('advisor_model.pkl is missing. Run: python train_model.py')
with MODEL_PATH.open('rb') as f:
    model = pickle.load(f)

ADVICE = {
    'Good standing': ('You appear to be in good academic standing.', [
        'Keep a consistent weekly study timetable.',
        'Review lecture notes within 24 hours of each class.',
        'Maintain attendance and begin exam revision early.'
    ]),
    'Needs improvement': ('Your results suggest that some areas need attention.', [
        'Create a realistic weekly study plan and prioritize difficult courses.',
        'Attend classes consistently and ask lecturers questions early.',
        'Use past questions for practice and join a focused study group.'
    ]),
    'At risk': ('Your inputs suggest that you may need additional academic support.', [
        'Contact your academic adviser or course lecturer for a support plan.',
        'Identify failed or difficult courses and schedule regular revision sessions.',
        'Improve attendance and set small, measurable weekly study goals.'
    ])
}

@app.route('/', methods=['GET', 'POST'])
def home():
    result = None
    if request.method == 'POST':
        try:
            cgpa = float(request.form['cgpa'])
            attendance = float(request.form['attendance'])
            study_hours = float(request.form['study_hours'])
            failed_courses = int(request.form['failed_courses'])
            if not (0 <= cgpa <= 5 and 0 <= attendance <= 100 and 0 <= study_hours <= 100 and 0 <= failed_courses <= 30):
                raise ValueError
            prediction = model.predict([[cgpa, attendance, study_hours, failed_courses]])[0]
            summary, tips = ADVICE[prediction]
            result = {'label': prediction, 'summary': summary, 'tips': tips}
        except (ValueError, KeyError):
            result = {'error': 'Enter valid values: CGPA 0–5, attendance 0–100%, study hours 0–100, and failed courses 0–30.'}
    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
