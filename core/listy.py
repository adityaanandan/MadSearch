import pandas as pd
import numpy as np
import sklearn

temp = pd.read_csv("./data/all_apartments.csv")
temp1 = pd.read_csv("./data/distance2.csv")
temp["distance"] = temp1['distance']

df = temp[['distance', 'rent $', 'Bedroom(s)']]
df = df.rename(columns={'rent $': 'rent', 'Bedroom(s)': 'bedrooms', 'Location': 'location'})
df['rent'] = df.rent.str.extract('\$(\d+)', expand=False)

df.rent = df.rent.astype(float)

def convert_bedrooms_to_float(bedroom_str):
    bedroom_str = bedroom_str.lower()
    if 'studio' in bedroom_str:
        return 1.0
    else:
        first_part = bedroom_str.split(',')[0]
        number_str = first_part.split(' ')[0]
        match = pd.to_numeric(number_str, errors='coerce')
        return match

df['bedrooms'] = df['bedrooms'].apply(convert_bedrooms_to_float)
df_train = df[["distance","rent", "bedrooms"]]



def search(df, location, bedrooms, max_rent):
  if location >= 5 and location <= 7:
    temp = df[df['distance'] <= 1.7]
    temp = temp[temp["bedrooms"] == bedrooms]
    temp = temp[temp['rent'] <= max_rent]
  elif location <= 10:
    temp = df[df['distance'] <= 1.25]
    temp = temp[temp["bedrooms"] == bedrooms]
    temp = temp[temp['rent'] <= max_rent]
  else: 
    temp = df[df["bedrooms"] == bedrooms]
    temp = temp[temp['rent'] <= max_rent]
  if len(temp) == 0: 
    return df_train.sample() 
  return temp.sample()




numerical_features = ['distance', 'rent', 'bedrooms']

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df_train[numerical_features] = scaler.fit_transform(df_train[numerical_features])

from sklearn.neighbors import NearestNeighbors
weights = {'distance': 1, 'rent': 2, 'bedrooms': 1}

scaler = StandardScaler()
df_train_scaled = scaler.fit_transform(df_train)

df_train_scaled = pd.DataFrame(df_train_scaled, columns=df_train.columns)

for column, weight in weights.items():
    df_train_scaled[column] *= weight

nn_model = NearestNeighbors(metric='euclidean')
nn_model.fit(df_train_scaled)


def recommend_similar_houses(query_house_index, num_recommendations=5):
    distances, indices = nn_model.kneighbors(df_train.iloc[query_house_index].values.reshape(1, -1), n_neighbors=num_recommendations+1)
    recommended_indices = indices.squeeze()[1:]
    interested_house  =  df.iloc[[query_house_index]].dropna(axis=1)
    recommended_houses = df.iloc[recommended_indices]

    return interested_house, recommended_houses


def app(budget, bedrooms, distance):
    i, r = recommend_similar_houses(search(df, distance, bedrooms, budget).index[0])
    appy = pd.concat([i, r], ignore_index=False, sort=False)
    total = pd.DataFrame(columns = temp.columns)
    count = 0
    for idx in appy.index:
        total.loc[count] = temp.loc[idx]
        count += 1

    return total









