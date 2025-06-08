import pandas as pd

# Load the datasets with ::: as separator, using raw strings
train_data = pd.read_csv(r'D:\1. PRAZZU\My Projects\Movie Genres\archive\Genre Classification Dataset\train_data.txt', sep=':::', engine='python', names=['ID', 'TITLE', 'GENRE', 'DESCRIPTION'])
test_data = pd.read_csv(r'D:\1. PRAZZU\My Projects\Movie Genres\archive\Genre Classification Dataset\test_data.txt', sep=':::', engine='python', names=['ID', 'TITLE', 'DESCRIPTION'])
test_solution = pd.read_csv(r'D:\1. PRAZZU\My Projects\Movie Genres\archive\Genre Classification Dataset\test_data_solution.txt', sep=':::', engine='python', names=['ID', 'TITLE', 'GENRE', 'DESCRIPTION'])

# Strip whitespace from column names and data
train_data.columns = train_data.columns.str.strip()
test_data.columns = test_data.columns.str.strip()
test_solution.columns = test_solution.columns.str.strip()
train_data = train_data.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
test_data = test_data.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
test_solution = test_solution.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

# Look at the first few rows
print("Train Data:")
print(train_data.head())
print("\nTest Data:")
print(test_data.head())
print("\nTest Solution:")
print(test_solution.head())

# Check the genres in train_data
print("\nGenre Counts:")
print(train_data['GENRE'].value_counts())