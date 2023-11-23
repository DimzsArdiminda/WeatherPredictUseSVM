import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix
from scipy.stats import mannwhitneyu, wilcoxon
from scipy.stats import mannwhitneyu

def weather_recommendation(weather_condition):
    if 'Fog' in weather_condition:
        return "Saran: Gunakan lampu kabut saat berkendara dan perhatikan kecepatan Anda."
    elif 'Freezing Drizzle' in weather_condition:
        return "Saran: Hindari berkendara jika tidak diperlukan. Jika berkendara, gunakan peralatan keselamatan musim dingin."
    else:
        return "Saran: Perhatikan kondisi cuaca dan sesuaikan aktivitas Anda."

# Load data into DataFrame
data = pd.read_excel("/content/drive/MyDrive/SVM/NewDataWeather.xlsx")
kolom = ['Temp_C', 'Dew Point Temp_C', 'Rel Hum_%', 'Wind Speed_km/h', 'Visibility_km', 'Pressure_kPa', 'Weather']

df = pd.DataFrame(data)

# Menghapus baris yang mengandung NaN
df_cleaned = df.dropna()

X = df_cleaned.drop('Weather', axis=1)

print("\nX variable\n\n", X.head())

y = df_cleaned['Weather']

print("\ny variable\n\n", y.head())
print("\n")

# Count plot
print("Count plots")
sns.countplot(x=df_cleaned['Weather'], data=df_cleaned, palette='hls')
plt.title("Weather count")
plt.show()
print("\n")

# Hypothesis Testing
# Non parametric test (Mann-Whitney test)
print("\n")
print("Hypothesis Testing\n")
print("Mann-Whitney Test results")
data1 = df_cleaned['Weather']
data2 = df_cleaned['Temp_C']

# Mengganti variabel kategorikal 'Weather' menjadi nilai numerik
df_cleaned['Weather_numeric'] = df_cleaned['Weather'].astype('category').cat.codes
# Melakukan uji Mann-Whitney U
stat, p = mannwhitneyu(df_cleaned['Weather_numeric'], df_cleaned['Temp_C'])
print("Statistics:", stat, "\n", "p-value:", p)
alpha = 0.05
if p > alpha:
    print('Same distribution (fail to reject H0)')
else:
    print('Different distribution (reject H0 and accepting H1)')

# Wilcoxon test
print("\n")
print("Wilcoxon Test Results")
data1 = df_cleaned['Wind Speed_km/h']
data2 = df_cleaned['Visibility_km']
stat1, p1 = wilcoxon(data1, data2)
print("Statistics:", stat1, "\n", "p-value:", p1)
# interpret the results
alpha = 0.05
if p1 > alpha:
    print('Same distribution (fail to reject H0)')
else:
    print('Different distribution (reject H0 and accepting H1)')
print("\n")

# Split the data train and test
print("Support Vector Machine Results\n")
# Split the data into train and validation sets (80:20)
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=0)

# Support Vector Machine classifier
svclassifier = SVC(kernel='linear')

# Fit the model on the training data
svclassifier.fit(X_train, y_train)

# Predict on the validation set
y_pred_valid = svclassifier.predict(X_valid)

# Evaluating the algorithm on the validation set
print("Validation Set Results\n")
print(classification_report(y_valid, y_pred_valid, zero_division=1))
print(confusion_matrix(y_valid, y_pred_valid))
print("\n")
accuracy_valid = metrics.accuracy_score(y_valid, y_pred_valid)
print("Accuracy score on validation set:", round(accuracy_valid, 2))



# print("masukan harus berupa boolean")
# # User input
# user_input = {
#     'Temp_C': float(input("Masukkan nilai Temp_C: ")),
#     'Dew Point Temp_C': float(input("Masukkan nilai Dew Point Temp_C: ")),
#     'Rel Hum_%': float(input("Masukkan nilai Rel Hum_%: ")),
#     'Wind Speed_km/h': float(input("Masukkan nilai Wind Speed_km/h: ")),
#     'Visibility_km': float(input("Masukkan nilai Visibility_km: ")),
#     'Pressure_kPa': float(input("Masukkan nilai Pressure_kPa: "))
# }

# # Buat DataFrame dari input pengguna
# user_input_data = pd.DataFrame([user_input])


# # Dapatkan nama fitur yang digunakan pada saat pelatihan
# trained_feature_names = svclassifier.support_


# # Sesuaikan nama fitur input pengguna dengan nama fitur yang digunakan pada saat pelatihan
# user_input_data = user_input_data.rename(columns={'Pressure_kPa': 'Pressure_kPa'})

# # Pastikan semua nama fitur ada di DataFrame input pengguna
# missing_features = set(trained_feature_names) - set(user_input_data.columns)

# # Inisialisasi user_input_data dengan nilai 0 untuk fitur yang tidak ada pada saat pelatihan
# for feature in missing_features:
#     user_input_data[feature] = 0

# # Konversi nama fitur menjadi string
# user_input_data.columns = user_input_data.columns.astype(str)

# # Prediksi cuaca berdasarkan input pengguna
# predicted_weather = svclassifier.predict(user_input_data)


# # Print predicted weather
# print("\nPredicted Weather:", predicted_weather[0])

# # Berikan rekomendasi cuaca
# recommendation = weather_recommendation(predicted_weather)
# print(recommendation)
