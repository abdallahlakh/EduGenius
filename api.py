from flask import Flask, request, jsonify
import joblib
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = joblib.load('logistic_regression_model.pkl')

# Function to transform data into PCA features
def transform_to_pca(data):
    try:
        # Create a DataFrame with two repeated samples
        df = pd.DataFrame([data, data])  # Duplicate the input sample
        print("df:\n", df)

        pca_features = []

        # List of categories to process
        categories = ['Programming_Languages', 'Frameworks', 'Databases', 'DevOps_Tools']
        scaler = StandardScaler()

        # Create a mapping for category name to the format for PCA components
        category_to_prefix = {
            'Programming_Languages': 'pro',
            'Frameworks': 'fr',
            'Databases': 'db',
            'DevOps_Tools': 'dv'
        }

        for category in categories:
            # Encode the category and handle PCA
            df_encoded = df[category].str.get_dummies(sep=', ')
            elements = list(df_encoded.columns)
            print(f"df_encoded for {category}:\n", df_encoded)

            if len(elements) > 0:
                # Standardize and apply PCA
                pca_input_scaled = scaler.fit_transform(df_encoded)
                n_components = min(2, len(elements), len(df) - 1)  # Ensure max 2 components
                pca = PCA(n_components=n_components)
                pca_result = pca.fit_transform(pca_input_scaled)

                # Ensure we generate both PC1 and PC2 components for each category
                pca_columns = [f'PC{i+1}_{category_to_prefix[category]}' for i in range(2)]
                pca_result = np.concatenate([pca_result, np.zeros((pca_result.shape[0], 2 - pca_result.shape[1]))], axis=1)

                pca_df = pd.DataFrame(
                    pca_result, 
                    columns=pca_columns
                )
                print(f"PCA result for {category}:\n", pca_df)
                pca_features.append(pca_df)
            else:
                # If no valid elements, add empty components
                empty_df = pd.DataFrame(
                    [[0, 0]] * len(df),
                    columns=[f'PC1_{category_to_prefix[category]}', f'PC2_{category_to_prefix[category]}']
                )
                print(f"No valid elements for {category}, adding empty PCA components.")
                pca_features.append(empty_df)

        # Concatenate all PCA features into a single DataFrame
        final_pca_df = pd.concat(pca_features, axis=1)
        print("Final PCA DataFrame:\n", final_pca_df)
        return final_pca_df

    except Exception as e:
        print(f"Error during PCA transformation: {str(e)}")
        return None

@app.route('/')
def home():
    return "Logistic Regression Model API"

# API endpoint for making predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the JSON data from the request
        data = request.get_json()

        # Transform the data into PCA features
        pca_features = transform_to_pca(data)

        if pca_features is None or pca_features.empty:
            raise ValueError("PCA transformation failed or produced no features.")

        # Extract the other features (Happy_in_Current_Role and Current_Role_Difficulty)
        rest_of_features = pd.DataFrame(
            [[data['Happy_in_Current_Role'], data['Current_Role_Difficulty']]],
            columns=['Happy_in_Current_Role', 'Current_Role_Difficulty']
        )
        print("Other features:\n", rest_of_features)

        # Concatenate PCA features with the other features
        final_features = pd.concat([pca_features.iloc[:1], rest_of_features], axis=1)
        print("Final features for prediction:\n", final_features)

        # Make prediction using the final features
        prediction = model.predict(final_features)
        job_roles = np.array([
            'Mobile Developer', 'Front-End Developer', 'Cloud Engineer',
            'Database Administrator', 'Data Scientist', 'Robotics Engineer',
            'Full-Stack Developer', 'Project Manager', 'DevOps Engineer',
            'Systems Analyst', 'AI Researcher', 'Network Engineer',
            'Game Developer', 'Machine Learning Engineer', 'Software Engineer',
            'Product Manager', 'IT Support Specialist',
            'Other (please specify)', 'Cybersecurity Analyst', 'Data Engineer',
            'Back-End Developer'
        ])
        programing_languages=['C', 'C#', 'C++', 'Dart', 'Go', 'Java',
       'JavaScript', 'Kotlin', 'PHP', 'Python', 'Ruby', 'Rust', 'Swift',
       'TypeScript']
        # Assuming 'df' is your 
        frameworks =['.NET Core', 'Angular', 'Apache Spark', 'Burp Suite',
         'Dask', 'Django', 'Electron', 'Express.js', 'Flask',
        'Flutter', 'GTK', 'Godot', 'JavaFX', 'Keras',
        'Kotlin Multiplatform Mobile (KMM)', 'Nestjs', 'Next.js',
        'Numpy', 'OpenVAS', 'Pandas', 'PyTorch', 'Qt', 'React', 'React Native',
        'Scikit-Learn', 'Snort', 'Svelte', 'SwiftUI',
        'TensorFlow', 'Unity', 'Vue.js', 'WPF', 'Wireshark',
        'Xamarin']
        devops=['AWS', 'Ansible', 'Apache', 'Azure', 'CircleCI', 'DigitalOcean',
       'Docker', 'Git', 'GitLab CI/CD', 'Google Cloud Platform', 'Jenkins',
       'Kubernetes', 'Nginx', 'Terraform', 'Travis CI']
        databases=['CassandraDB', 'Microsoft SQL Server', 'MongoDB', 'MySQL', 'OracleDB',
       'PostgreSQL', 'Redis', 'SQLite']


        # Convert prediction to the job role name
        predicted_job_role = job_roles[int(prediction[0])]

        # Return the prediction as JSON
        return jsonify({'predicted_current_role': predicted_job_role})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
