from flask import Flask, request, jsonify
import numpy as np
import joblib

app = Flask(__name__)
model = joblib.load('framework_model.pkl')
roles = [
    'Systems Analyst', 'Mobile Developer', 'Cybersecurity Analyst', 'Data Scientist',
    'DevOps Engineer', 'Front-End Developer', 'Network Engineer', 'IT Support Specialist',
    'Product Manager', 'Software Engineer', 'Game Developer', 'Other (please specify)',
    'AI Researcher', 'Project Manager', 'Robotics Engineer', 'Full-Stack Developer',
    'Cloud Engineer', 'Back-End Developer', 'Machine Learning Engineer',
    'Database Administrator', 'Data Engineer'
]
# Conversion functions    
def construct_year_index(x):
    if x == '1-3 years':
        return 1 
    if x == 'Less than 1 year':
        return 2
    if x == 'More than 5 years':
        return 3
    if x == '3-5 years':
        return 4
    return 5

def construct_language_index(x):
    if x == 'C++':
        return 1
    if x == 'Python':
        return 2
    if x == 'Java':
        return 3
    return 4

def construct_database_index(x):
    if x == 'MongoDB':
        return 1
    if x == 'PostgreSQL':
        return 2
    if x == 'MySQL':
        return 3
    return 4

def construct_framework_index(x):
    if x == 'React':
        return 1
    if x == 'Django':
        return 2
    if x == 'TensorFlow':
        return 3
    return 4

def construct_devops_index(x):
    if x == 'Git':
        return 1
    if x == 'Docker':
        return 2
    if x == 'AWS':
        return 3
    return 4

# Function to convert 'Yes'/'No' to 1/0
def yes_no_to_binary(x):
    if x == 'Yes':
        return 1
    return 0

# Route to handle POST requests
@app.route('/predict', methods=['POST'])
def predict_role():
    data = request.json
    try:
        # Parse the JSON request
        year_index = construct_year_index(data.get('Years_in_CS'))
        language_index = construct_language_index(data.get('Programming_Languages'))
        database_index = construct_database_index(data.get('Databases'))
        framework_index = construct_framework_index(data.get('Frameworks'))
        devops_index = construct_devops_index(data.get('DevOps_Tools'))

        # Convert categorical 'Yes'/'No' to binary (1/0)
        networking_experience = yes_no_to_binary(data.get('Networking_Experience'))
        cybersecurity_experience = yes_no_to_binary(data.get('Cybersecurity_Experience'))
        ai_data_science_experience = yes_no_to_binary(data.get('AI_Data_Science_Experience'))
        happy_in_current_role = yes_no_to_binary(data.get('Happy_in_Current_Role'))
        current_role_difficulty = yes_no_to_binary(data.get('Current_Role_Difficulty'))

        features = [year_index, language_index, database_index, framework_index, devops_index, 
                    networking_experience, cybersecurity_experience, ai_data_science_experience, 
                    happy_in_current_role, current_role_difficulty]

        input_data = np.array(features).reshape(1, -1)
        
        # Perform prediction
        prediction_index = model.predict(input_data)[0]
        
        # Get the predicted role from the list using the index
        predicted_role = roles[prediction_index]  # Use the prediction index to get the corresponding role
        
        # Return the prediction as a JSON response
        response = {
            "predicted_current_role": predicted_role
        }
        return jsonify(response), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400
    
if __name__ == '__main__':
    app.run(debug=True)


# PS C:\Users\laab2\OneDrive\Bureau\k-111> $headers = @{"Content-Type" = "application/json"}; $body = '{    "Years_in_CS": "3-5 years",     "Programming_Languages": "Python",     "Databases": "PostgreSQL",     "Frameworks": "React",     "DevOps_Tools": "Docker",     "Networking_Experience": "Yes", 
#     "Cybersecurity_Experience": "No",     "AI_Data_Science_Experience": "Yes",     "Happy_in_Current_Role": "Yes",     "Current_Role_Difficulty": "No"}'; Invoke-WebRequest -Uri http://127.0.0.1:5000/predict -Method POST -Headers $headers -Body $body