import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv("cleaned_data.txt")

# Make test and train data
train, test = train_test_split(data, test_size=0.2)

train.to_csv(r'cleaned_train_data.txt', index=False)
test.to_csv(r'cleaned_test_data.txt', index=False)

# Make small data
train, test = train_test_split(data, test_size=0.05)

test.to_csv(r'cleaned_small_data.txt', index=False)

# Make small test and train data
small_data = pd.read_csv("cleaned_small_data.txt")

train, test = train_test_split(test, test_size=0.2)

train.to_csv(r'cleaned_small_train_data.txt', index=False)
test.to_csv(r'cleaned_small_test_data.txt', index=False)
