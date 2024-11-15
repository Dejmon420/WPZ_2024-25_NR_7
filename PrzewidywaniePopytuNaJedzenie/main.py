# Importowanie bibliotek
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Wczytywanie danych
train_data = pd.read_csv('train.csv')
center_info = pd.read_csv('fulfilment_center_info.csv')
meal_info = pd.read_csv('meal_info.csv')
test_data = pd.read_csv('test_QoiMO9B.csv')

# Łączenie danych
train_data = pd.concat([train_data, test_data], axis=0)
train_data = train_data.merge(center_info, on='center_id', how='left')
train_data = train_data.merge(meal_info, on='meal_id', how='left')

# Inżynieria cech
train_data['discount_amount'] = train_data['base_price']-train_data['checkout_price']
train_data['discount_percentage'] = (train_data['discount_amount'] / train_data['base_price']) * 100
train_data['discount_y/n'] = (train_data['base_price'] > train_data['checkout_price']).astype(int)

# Kodowanie zmiennych kategorycznych
train_data = pd.get_dummies(train_data, columns=['center_type', 'category', 'cuisine'], drop_first=True)

# Podział na zbiór treningowy i testowy
train = train_data[train_data['week'].isin(range(1, 146))]
test = train_data[train_data['week'].isin(range(146, 156))]

# Selekcja cech i etykiety
x_train = train.drop(['num_orders', 'id', 'week'], axis=1)
y_train = train['num_orders']
x_test = train.drop(['num_orders', 'id', 'week'], axis=1)

# Trenowanie modelu RandomForestRegressor
rf_model = RandomForestRegressor(n_estimators=1000, random_state=42)
rf_model.fit(x_train, y_train)
rf_predictions = rf_model.predict(x_test)

# Trenowanie modelu GradientBoostingRegressor
gb_model = GradientBoostingRegressor(n_estimators=1000, random_state=42)
gb_model.fit(x_train, y_train)
gb_predictions = gb_model.predict(x_test)

# Ocena modeli na zbiorze treningowym
rf_train_predictions = rf_model.predict(x_train)
gb_train_predictions = gb_model.predict(x_train)

rf_rmse = mean_squared_error(y_train, rf_train_predictions, squared=False)
gb_rmse = mean_squared_error(y_train, gb_train_predictions, squared=False)

# Wizualizacja wynikóww RMSE dal każdego modelu
plt.figure(figsize=(10, 5))
plt.bar(['Random Forest', 'Gradient Boosting'], [rf_rmse, gb_rmse], color=['blue', 'green'])
plt.ylabel('RMSE')
plt.title('Porównanie błędu RMSE dla Random Forest i Gradient Boosting')
plt.show()

# Wizualizacja porównania wartości rzeczywistych i przewidywanych
# Używamy tylko próbek z testu dla czytelności wykresu
plt.figure(figsize=(12, 6))
plt.plot(y_train.values[:50], label='Rzeczywiste', color='black')
plt.plot(rf_predictions[:50], label='Random Forest Przewidywania', color='blue')
plt.plot(gb_predictions[:50], label='Gradient Boosting Przewidywania', color='green')
plt.legend()
plt.xlabel('Indeks próby')
plt.ylabel('Liczba zamówień')
plt.title('Porównanie rzeczywistych i przewidywanych wartości liczby zamówień')
plt.show()