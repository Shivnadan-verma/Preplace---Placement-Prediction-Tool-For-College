from flask import Flask, render_template, request, redirect, url_for, flash, session
from flask_mysqldb import MySQL
from werkzeug.security import generate_password_hash, check_password_hash
import numpy as np
import pandas as pd
import pickle
import os

# Initialize Flask app
app = Flask(__name__)

# Flask-MySQLdb Configuration for `flask_app`
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = 'abc'  
app.config['MYSQL_DB'] = 'flask_app'
mysql = MySQL(app)

# Secret key for session
app.secret_key = 'your_secret_key'  # Replace with a strong secret key

# Set up static folder for output files
OUTPUT_FOLDER = 'static'
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

# Load the trained model and scaler
try:
    with open('final_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)

    with open('final_scaler.pkl', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
except FileNotFoundError as e:
    raise RuntimeError(f"Model or scaler file not found: {e}")

# Routes
@app.route('/')
def home():
    return render_template('main.html')

@app.route('/ourmission')
def ourmission():
    return render_template('ourmission.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm-password']

        if password != confirm_password:
            flash("Passwords do not match!", "danger")
            return redirect(url_for('register'))

        cursor = mysql.connection.cursor()
        cursor.execute("SELECT * FROM users WHERE email = %s", (email,))
        existing_user = cursor.fetchone()

        if existing_user:
            flash("Email already registered!", "danger")
            return redirect(url_for('login'))

        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
        cursor.execute("INSERT INTO users (name, email, password) VALUES (%s, %s, %s)",
                       (name, email, hashed_password))
        mysql.connection.commit()
        cursor.close()

        flash("Registration successful! Please log in.", "success")
        return redirect(url_for('login'))

    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'user_email' in session:
        return redirect(url_for('landing'))

    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        cursor = mysql.connection.cursor()
        cursor.execute("SELECT * FROM users WHERE email = %s", (email,))
        user = cursor.fetchone()
        cursor.close()

        if user and check_password_hash(user[3], password):  # user[3] is the password field
            session['user_email'] = user[1]  # user[1] is the email field
            flash("Login successful!", "success")
            return redirect(url_for('landing'))
        else:
            flash("Incorrect email or password.", "danger")

    return render_template('login.html')

@app.route('/landing')
def landing():
    if 'user_email' not in session:
        flash("Please log in first.", "warning")
        return redirect(url_for('login'))
    return render_template('landing.html')

@app.route('/chatbot')
def chatbot():
    return render_template('chatbot.html')  # This serves chatbot.html

@app.route('/prediction', methods=['POST'])
def prediction():
    if 'file' in request.files:
        file = request.files['file']
        if file.filename.endswith('.csv'):
            try:
                df = pd.read_csv(file)
                required_columns = [
                    'sessional_marks', 'Class_attendance', 'project_marks',
                    'LA2SS', 'LA3SS', 'LA4SS', 'LA5SS', 'LA6SS',
                    'QA2SS', 'QA3SS', 'QA4SS', 'QA5SS', 'QA6SS',
                    'EC2SS', 'EC3SS', 'EC4SS', 'EC5SS', 'EC6SS',
                    'DS3S', 'DS4S', 'DS5S', 'DS6S',
                    'AS5S', 'AS6S'
                ]
                missing_columns = [col for col in required_columns if col not in df.columns]

                if missing_columns:
                    return render_template('index.html', prediction_text=f"Missing required columns: {missing_columns}")

                df.fillna(df.mean(numeric_only=True), inplace=True)

                df['Logical_Ability'] = df[['LA2SS', 'LA3SS', 'LA4SS', 'LA5SS', 'LA6SS']].mean(axis=1)
                df['Quantitative_Ability'] = df[['QA2SS', 'QA3SS', 'QA4SS', 'QA5SS', 'QA6SS']].mean(axis=1)
                df['English_Comprehension'] = df[['EC2SS', 'EC3SS', 'EC4SS', 'EC5SS', 'EC6SS']].mean(axis=1)
                df['Domain_Knowledge'] = df[['DS3S', 'DS4S', 'DS5S', 'DS6S']].mean(axis=1)
                df['Automata_Programming'] = df[['AS5S', 'AS6S']].mean(axis=1)

                X = df[['sessional_marks', 'Class_attendance', 'Logical_Ability',
                        'Quantitative_Ability', 'English_Comprehension', 
                        'Domain_Knowledge', 'Automata_Programming', 'project_marks']]

                X_scaled = scaler.transform(X)
                predictions = model.predict(X_scaled)
                df['placement_status'] = ['Strong Chance' if p == 1 else 'Minimal Chance' for p in predictions]

                output_file = os.path.join(OUTPUT_FOLDER, 'predicted_placements.csv')
                df.to_csv(output_file, index=False)

                # Status counts and percentages
                status_counts = df['placement_status'].value_counts()
                total_students = len(df)
                strong_chance_percentage = round((status_counts.get('Strong Chance', 0) / total_students) * 100, 2)
                minimal_chance_percentage = round((status_counts.get('Minimal Chance', 0) / total_students) * 100, 2)

                graph_data = {
                    'labels': ['Strong Chance', 'Minimal Chance'],
                    'values': [int(status_counts.get('Strong Chance', 0)), int(status_counts.get('Minimal Chance', 0))],
                    'percentages': [strong_chance_percentage, minimal_chance_percentage]
                }

                return render_template('index.html',
                                       prediction_text=(
                                           f"Prediction completed successfully! "
                                           f"{strong_chance_percentage}% students have a Strong Chance, "
                                           f"{minimal_chance_percentage}% have a Minimal Chance."
                                       ),
                                       download_link=f"/static/predicted_placements.csv",
                                       graph_data=graph_data)

            except Exception as e:
                return render_template('index.html', prediction_text=f"Error: {str(e)}")
        else:
            return render_template('index.html', prediction_text="Please upload a valid CSV file.")
    else:
        try:
            form_data = []
            required_fields = [
                'sessional_marks', 'Class_attendance', 'Logical_Ability',
                'Quantitative_Ability', 'English_Comprehension', 'Domain_Knowledge',
                'Automata_Programming', 'project_marks'
            ]

            for key in required_fields:
                value = request.form.get(key, '').strip()
                if not value.isdigit() and not value.replace('.', '', 1).isdigit():
                    raise ValueError(f"Invalid input for {key}: {value}")
                form_data.append(float(value))

            input_data_scaled = scaler.transform([form_data])
            prediction = model.predict(input_data_scaled)
            placement_status = "Strong Chance" if prediction[0] == 1 else "Minimal Chance"

            return render_template('index.html', prediction_text=f"The student has {placement_status}.")
        except ValueError as ve:
            return render_template('index.html', prediction_text=f"Input error: {str(ve)}")
        except Exception as e:
            return render_template('index.html', prediction_text=f"Error: {str(e)}")
        
@app.route('/api/getPlacementData', methods=['GET'])
def get_placement_data():
    try:
        # Query the database for placement data
        cursor = mysql.connection.cursor()
        cursor.execute("SELECT placement_status, COUNT(*) AS count FROM students GROUP BY placement_status")
        placement_data = cursor.fetchall()
        cursor.close()

        total_students = sum(row[1] for row in placement_data)
        placed_count = next((row[1] for row in placement_data if row[0] == 'Strong Chance'), 0)
        not_placed_count = next((row[1] for row in placement_data if row[0] == 'Minimal Chance'), 0)

        # Example hardcoded chart data (replace with database queries as needed)
        chart_data = {
            "cgpa": [placed_count, not_placed_count],  # Replace with actual CGPA data
            "internships": [60, 40]  # Replace with actual internship data
        }

        return {
            "totalStudents": total_students,
            "placed": placed_count,
            "notPlaced": not_placed_count,
            "chartData": chart_data
        }
    except Exception as e:
        return {"error": str(e)}, 500


@app.route('/logout')
def logout():
    session.pop('user_email', None)
    flash("You have been logged out.", "info")
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)
