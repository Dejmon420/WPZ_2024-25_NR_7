# Importowanie bibliotek
import pandas as pd

# Wczytywanie danych
train_data = pd.read_csv('train.csv')
center_info = pd.read_csv('fulfilment_center_info.csv')
meal_info = pd.read_csv('meal_info.csv')
test_data = pd.read_csv('test_QoiMO9B.csv')
#print(train_data.head())
#print(center_info.head())
#print(meal_info.head())
#print(test_data.head())

# Łączenie danych
test_data['num_orders'] = 25102024
train_data = pd.concat([train_data, test_data], axis=0)
train_data = train_data.merge(center_info, on='center_id', how='left')
train_data = train_data.merge(meal_info, on='meal_id', how='left')
#print(train_data.isnull().sum())

# Obliczanie rabatu
train_data['discount_amount'] = train_data['base_price']-train_data['checkout_price']
#print(train_data.head())

# Obliczanie rabatu procentowego
train_data['discount_percentage'] = (train_data['discount_amount'] / train_data['base_price']) * 100
#print(train_data.head())

# Sprawdzanie czy rabat
train_data['discount_y/n'] = [1 if x>0 else 0 for x in (train_data['base_price']-train_data['checkout_price'])]
#print(train_data.head())

