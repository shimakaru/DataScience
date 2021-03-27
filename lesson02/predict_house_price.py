import os
import pandas as pd
#import keras

def load_housing_data(housing_path="datasets/housing/"):
	csv_path=os.path.join(housing_path,"housing.csv")
	return pd.read_csv(csv_path)


housing=load_housing_data()
print(housing.head())