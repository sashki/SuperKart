import pandas as pd
import joblib
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load model, scaler, and training columns
model = joblib.load('model/superkart_model.joblib')
scaler = joblib.load('model/superkart_scaler.joblib')
training_columns = joblib.load('model/superkart_columns.joblib')  # <-- added this

print("Model, scaler, and training columns loaded.")

def preprocess_input(data):
    df = pd.DataFrame([data])
    df['Store_Age'] = 2025 - df['Store_Age_Years']
    df = df.drop(columns=['Product_Id_char', 'Store_Age_Years'])

    categorical_cols = ['Product_Sugar_Content', 'Store_Size', 'Store_Location_City_Type', 'Store_Type', 'Product_Type_Category']
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # Align with training columns
    for col in training_columns:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
    df_encoded = df_encoded[training_columns]

    # Scale numerical features
    num_cols_to_scale = ['Product_Weight', 'Product_Allocated_Area', 'Product_MRP', 'Store_Age']
    df_encoded[num_cols_to_scale] = scaler.transform(df_encoded[num_cols_to_scale])
    return df_encoded

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = request.get_json()
        processed = preprocess_input(input_data)
        pred = model.predict(processed)
        return jsonify({'predicted_sales': float(pred[0])})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'message': 'SuperKart Sales Prediction API',
        'endpoints': {
            '/health': 'GET - Check if API is running',
            '/predict': 'POST - Send JSON data to get sales prediction'
        }
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)