import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix

#saran
def weather_recommendation(weather_condition):
    if 'Fog' in weather_condition:
        return "Hindari berkendara saat kabut tebal, gunakan lampu kabut, dan jaga jarak aman antara kendaraan."
    elif 'Freezing' in weather_condition:
        return "Pastikan kendaraan Anda dilengkapi dengan ban salju atau rantai ban salju, dan hindari berkendara di jalan yang licin."
    elif 'Drizzle' in weather_condition:
        return "Pastikan Anda membawa payung atau jas hujan saat bepergian."
    elif 'Mostly Cloudy' in weather_condition or 'Cloudy' in weather_condition:
        return "Pastikan Anda membawa jaket atau pakaian hangat saat bepergian."
    elif 'Rain' in weather_condition or 'Rain Showers' in weather_condition:
        return "Pastikan Anda membawa payung atau jas hujan saat bepergian."
    elif 'Mainly Clear' in weather_condition or 'Clear' in weather_condition:
        return "Pastikan Anda membawa kacamata hitam dan tabir surya saat bepergian."
    elif 'Freezing Rain' in weather_condition:
        return "Hindari berkendara di jalan yang licin, pastikan kendaraan Anda dilengkapi dengan ban salju atau rantai ban salju, dan gunakan lampu kabut saat berkendara."
    elif 'Snow Showers' in weather_condition or 'Snow' in weather_condition:
        return "Pastikan kendaraan Anda dilengkapi dengan ban salju atau rantai ban salju, dan hindari berkendara di jalan yang licin."
    elif 'Thunderstorms' in weather_condition:
        return "Hindari berada di luar ruangan saat petir dan pastikan Anda berada di tempat yang aman."
    else:
        return "Saran: Perhatikan kondisi cuaca dan sesuaikan aktivitas Anda."

# Load data into DataFrame
data = pd.read_excel(r'/content/gdrive/My Drive/Tugas/Book8.xlsx')
kolom = ['Temp_C', 'Dew Point Temp_C', 'Rel Hum_%', 'Wind Speed_km/h', 'Visibility_km', 'Pressure_kPa', 'Weather']

df = pd.DataFrame(data)

# Menghapus baris yang mengandung NaN
df_cleaned = df.dropna()

# Split the data into train and validation sets (80:20)
X_train, X_valid, y_train, y_valid = train_test_split(df_cleaned.drop('Weather', axis=1), df_cleaned['Weather'], test_size=0.2, random_state=0)

# Load the standard scaler used during training
scaler = StandardScaler().fit(X_train)

# Transform the training and validation data
X_train_scaled = scaler.transform(X_train)
X_valid_scaled = scaler.transform(X_valid)

# Hyperparameter tuning using GridSearchCV
param_grid = {'C': [1, 10, 100], 'gamma': [0.1, 1, 10], 'kernel': ['rbf']}
grid = GridSearchCV(SVC(class_weight='balanced'), param_grid, refit=True, verbose=2, cv=3)
grid.fit(X_train_scaled, y_train)

# Print the best parameters found by GridSearchCV
print("Best Parameters:", grid.best_params_)

# Predict on the validation set using the best parameters
y_pred_valid_tuned = grid.predict(X_valid_scaled)

# Evaluating the algorithm on the validation set after tuning
print("\nValidation Set Results after Tuning\n")
print(classification_report(y_valid, y_pred_valid_tuned, zero_division=1))
print(confusion_matrix(y_valid, y_pred_valid_tuned))
print("\n")
accuracy_valid_tuned = metrics.accuracy_score(y_valid, y_pred_valid_tuned)
print("Accuracy score on validation set after tuning:", round(accuracy_valid_tuned, 2))

# User input
user_input = {
    'Temp_C': float(input("Masukkan nilai Temp_C: ")),
    'Dew Point Temp_C': float(input("Masukkan nilai Dew Point Temp_C: ")),
    'Rel Hum_%': float(input("Masukkan nilai Rel Hum_%: ")),
    'Wind Speed_km/h': float(input("Masukkan nilai Wind Speed_km/h: ")),
    'Visibility_km': float(input("Masukkan nilai Visibility_km: ")),
    'Pressure_kPa': float(input("Masukkan nilai Pressure_kPa: "))
}

# Create DataFrame from user input
user_input_data = pd.DataFrame([user_input])

# Adjust column names
user_input_data = user_input_data.rename(columns={'Pressure_kPa': 'Press_kPa'})

# Ensure all feature names are in the user input DataFrame
missing_features = set(X_train.columns) - set(user_input_data.columns)
for feature in missing_features:
    user_input_data[feature] = 0

# Convert column names to string
user_input_data.columns = user_input_data.columns.astype(str)

# Normalize the user input data using the same scaler
user_input_scaled = scaler.transform(user_input_data)

# Predict weather based on user input
predicted_weather = grid.predict(user_input_scaled)

# Print predicted weather
print("\nPredicted Weather:", predicted_weather[0])

# Provide weather recommendation
recommendation = weather_recommendation(predicted_weather)
print(recommendation)

