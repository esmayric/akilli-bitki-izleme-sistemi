from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import random
from sklearn.ensemble import IsolationForest
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Flask uygulamasını oluşturma ve CORS desteğini aktif etme
app = Flask(__name__)
CORS(app)

# Plants verilerini okuma
plants_df = pd.read_csv(
    r"C:\\Users\\horat\\Downloads\\pythonProject6v2\\pythonProject6\\backend\\plants_dataset_cleaned.csv",
    encoding='ISO-8859-1'  # Kodlamayı dosyanızın kodlamasına göre değiştirin
)

# Bitki adlarını ve ideal değerleri sözlük olarak oluşturma
plants_data = plants_df.set_index('Plant')[[
    'Min Lux', 'Max Lux', 'Min Temp', 'Max Temp',
    'Min moisture (Summer)', 'Max moisture (Summer)',
    'Min moisture (Winter)', 'Max moisture (Winter)', 'Soil pH'
]].to_dict('index')

# Sensör verilerini oluşturma ve Isolation Forest modelini buna göre eğitme
sensor_data = []

# Gerçekçi sensör verilerini üretme
for _ in range(100):
    sensor_data.append([
        random.randint(500, 10000),  # light
        random.randint(10, 40),      # temperature
        random.randint(10, 100),     # moisture
        round(random.uniform(5.0, 8.0), 1)  # pH
    ])

# Modeli eğitmek için sensör verisini numpy dizisine dönüştürme
sensor_data_np = np.array(sensor_data)

# Veriyi ölçeklendirme
scaler = StandardScaler()
sensor_data_scaled = scaler.fit_transform(sensor_data_np)

# Eğitim ve test verilerini ayırma
X_train, X_test = train_test_split(sensor_data_scaled, test_size=0.2, random_state=42)

# Isolation Forest modelini oluşturma ve parametre iyileştirme
model = IsolationForest(contamination=0.1, n_estimators=100, max_samples=256, random_state=42)
model.fit(X_train)

@app.route('/', methods=['GET'])
def home():
    """Ana sayfa rotası."""
    return jsonify({"message": "Welcome to the Plant Sensor API! Use /plants, /sensor, or /visualize endpoints."})

@app.route('/plants', methods=['GET'])
def get_plants():
    """Tüm bitki adlarını döndürür."""
    return jsonify(list(plants_data.keys()))

@app.route('/sensor', methods=['POST'])
def get_sensor_data():
    """Sensörden gelen verilerle bitki önerilerini döndürür."""
    data = request.get_json()
    flower = data.get('flower')
    season = data.get('season', 'Summer')  # Varsayılan olarak yaz seçilir

    if flower in plants_data:
        # Sensör verilerini simüle etme
        sensor_light = random.randint(100, 10000)
        sensor_temp = random.randint(10, 40)
        sensor_moisture = random.randint(10, 100)
        sensor_ph = round(random.uniform(5.0, 8.0), 1)

        # Sensör verilerini anomali tespiti için modele gönderme
        sensor_values = np.array([[sensor_light, sensor_temp, sensor_moisture, sensor_ph]])
        sensor_values_scaled = scaler.transform(sensor_values)
        anomaly_score = model.predict(sensor_values_scaled)[0]

        # Anomali tespiti sonucu
        if anomaly_score == -1:
            anomaly_status = "Anomaly detected in sensor data."
        else:
            anomaly_status = "Sensor data is normal."

        plant_data = plants_data[flower]

        # Mevsime göre nem aralıklarını seçme
        if season.lower() == 'summer':
            min_moisture = plant_data['Min moisture (Summer)']
            max_moisture = plant_data['Max moisture (Summer)']
        elif season.lower() == 'winter':
            min_moisture = plant_data['Min moisture (Winter)']
            max_moisture = plant_data['Max moisture (Winter)']
        else:
            return jsonify({"error": "Invalid season provided. Use 'Summer' or 'Winter'."}), 400

        # Sensör değerlerini ideal değerlerle karşılaştırma
        light_status = check_if_ideal(sensor_light, plant_data['Min Lux'], plant_data['Max Lux'])
        temp_status = check_if_ideal(sensor_temp, plant_data['Min Temp'], plant_data['Max Temp'])
        moisture_status = check_if_ideal(sensor_moisture, min_moisture, max_moisture)
        ph_status = check_if_ideal(sensor_ph, plant_data['Soil pH'] - 0.5, plant_data['Soil pH'] + 0.5)

        response_data = {
            "flower": flower,
            "season": season,
            "sensor_light": sensor_light,
            "ideal_light": [plant_data['Min Lux'], plant_data['Max Lux']],
            "light_status": light_status,
            "sensor_temp": sensor_temp,
            "ideal_temp": [plant_data['Min Temp'], plant_data['Max Temp']],
            "temp_status": temp_status,
            "sensor_moisture": sensor_moisture,
            "ideal_moisture": [min_moisture, max_moisture],
            "moisture_status": moisture_status,
            "sensor_ph": sensor_ph,
            "ideal_ph": [plant_data['Soil pH'] - 0.5, plant_data['Soil pH'] + 0.5],
            "ph_status": ph_status,
            "recommendation": f"Keep the {flower} in light between {plant_data['Min Lux']} and {plant_data['Max Lux']} lux.",
            "anomaly_status": anomaly_status
        }

        return jsonify(response_data)

    return jsonify({"error": "Bitki Bulunamadı"}), 404


def check_if_ideal(value, min_val, max_val):
    """Bir değerin ideal aralıkta olup olmadığını kontrol eder."""
    if value < min_val:
        return f"Too low! Min: {min_val}"
    elif value > max_val:
        return f"Too high! Max: {max_val}"
    else:
        return "Ideal"

if __name__ == '__main__':
    app.run(debug=True, port=5000)
