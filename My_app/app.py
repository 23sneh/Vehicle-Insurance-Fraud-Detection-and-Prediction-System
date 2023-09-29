from flask import Flask, render_template, request
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
import pandas as pd  # Import pandas

app = Flask(__name__)

# Load the trained XGBoost pipeline
with open('xgboost_fraud_detection_model.pkl', 'rb') as file:
    xgboost_pipeline = pickle.load(file)

# Create label encoders for categorical features
label_encoders = {
    'PolicyHolderOccupation': LabelEncoder(),
    'ClaimCause': LabelEncoder(),
    'AgeGroup': LabelEncoder(),
    'DrivingExperience': LabelEncoder()
}

categorical_features = ['PolicyHolderOccupation', 'ClaimCause', 'AgeGroup', 'DrivingExperience']

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        features = {}
        for feature in request.form:
            if request.form[feature]:  # Check if the field is not empty
                if feature in categorical_features:
                    encoder = label_encoders[feature]
                    encoder.fit_transform([request.form[feature]])  # Fit and transform on the fly
                    features[feature] = encoder.transform([request.form[feature]])[0]
                else:
                    features[feature] = float(request.form[feature]) if feature == 'ClaimAmount' else int(request.form[feature])

        # Apply preprocessing steps
        scaler = StandardScaler()
        pca = PCA(n_components=0.95)

        numerical_features = ['ClaimAmount', 'ClaimsFrequency', 'SafetyRating', 'PolicyHolderIncome', 'IncidentHourOfDay', 'witness', 'PastAccidents']
        scaled_numerical_features = scaler.fit_transform([[features.get(feat, 0) for feat in numerical_features]])
        scaled_features = dict(zip(numerical_features, scaled_numerical_features[0]))
        features.update(scaled_features)

        # Convert features to a pandas DataFrame
        features_df = pd.DataFrame([features])

        # Make prediction using the loaded pipeline
        prediction = xgboost_pipeline.predict(features_df)
        prediction_result = 'No' if prediction[0] == 0 else 'Yes'

        return render_template('result.html', prediction_result=prediction_result)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(port=9999, debug=True)
