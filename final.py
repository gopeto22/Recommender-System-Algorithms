import sqlite3
import numpy as np

"""
    Creates a SQLite database and two tables: 'train_ratings' and 'test_ratings'.
    Drops the tables if they already exist.
    Inserts data from CSV files into the tables.
    Closes the database connection.
    """
# Connect to the SQLite database
dataConnector = sqlite3.connect('recommender_system2.db')
loader = dataConnector.cursor()

# Drop the 'train_ratings' and 'test_ratings' tables if they already exist
loader.execute('DROP TABLE IF EXISTS train_ratings')
loader.execute('DROP TABLE IF EXISTS test_ratings')

# Create the 'train_ratings' and 'test_ratings' tables
loader.execute('''CREATE TABLE train_ratings (user_id INTEGER, item_id INTEGER, rating REAL, timestamp INTEGER)''')
loader.execute('''CREATE TABLE test_ratings (user_id INTEGER, item_id INTEGER, timestamp INTEGER)''')

# Commit the changes to the database
dataConnector.commit()

def insert_data_from_csv(file_path, table_name):
    with open(file_path, 'r') as f:
        if table_name == 'test_ratings':
            loader.executemany(f'''
                INSERT INTO {table_name} VALUES (?,?,?)
            ''', (line.strip().split(',')[:3] for line in f))  # Only take the first 3 values
        else:
            loader.executemany(f'''
                INSERT INTO {table_name} VALUES (?,?,?,?)
            ''', (line.strip().split(',') for line in f))

# Insert data into train_ratings
insert_data_from_csv('train_20M_withratings.csv', 'train_ratings')

# Insert data into test_ratings
insert_data_from_csv('test_20M_withoutratings.csv', 'test_ratings')


# Close the database connection
dataConnector.close()

def dataHandler(loader, dataSize):
    """
    A generator function that handles data loading and processing.

    Parameters:
    - loader: The data loader object.
    - dataSize: The number of data points to fetch at a time.

    Returns:
    - A numpy array containing the fetched data.

    """
    while True:
        # Fetch a batch of data.
        data = loader.fetchmany(dataSize) 
        if not data:
            # Exit loop if no data is fetched.
            break 
        # Yield batch as numpy array
        yield np.array(data, dtype=np.float32) 

def factoriseMatrix(dataFile, latentFactors, epochs=100, startAlpha=0.0035, startBeta=0.011, dataSize=180000, tolerance=0.0012):
    """
    Factorizes a matrix using collaborative filtering technique.

    Args:
        dataFile (str): The path to the data file.
        latentFactors (int): The number of latent factors.
        epochs (int, optional): The number of training epochs. Defaults to 100.
        startAlpha (float, optional): The initial learning rate. Defaults to 0.0035.
        startBeta (float, optional): The initial regularization parameter. Defaults to 0.011.
        dataSize (int, optional): The number of data samples to process per epoch. Defaults to 180000.
        tolerance (float, optional): The tolerance for early stopping. Defaults to 0.0012.

    Returns:
        tuple: A tuple containing the user matrix and item matrix.
    """
    # Connect to SQLite database.
    dataConnector = sqlite3.connect(dataFile)
    # Create a cursor object using the connection
    loader = dataConnector.cursor()
    # Fetching the maximum user_id and item_id from the train_ratings table
    loader.execute('SELECT MAX(user_id), MAX(item_id) FROM train_ratings')
    userMax, itemMax = loader.fetchone()
    # Compute size for item matrix.
    maximalItem = itemMax + 1 
    # Compute size for user matrix.
    maximalUser = userMax + 1  
    # Initializing user and item matrices with random values
    itemMatrix = np.random.normal(scale=1./latentFactors, size=(maximalItem, latentFactors))
    userMatrix = np.random.normal(scale=1./latentFactors, size=(maximalUser, latentFactors))
    # Initialize learning rate
    finalAlpha = startAlpha
    # Initialize MAE for early stopping
    finalMAE = float('inf')

    try:
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs} started")
            loader.execute('SELECT user_id, item_id, rating FROM train_ratings')
            # Iterating over the data and updating the user and item matrices
            for data in dataHandler(loader, dataSize):
                for userId, itemId, rating in data:
                    guess = np.dot(userMatrix[int(userId)], itemMatrix[int(itemId)])
                    wrongGuess = rating - guess
                    itemMatrix2 = finalAlpha * (wrongGuess * userMatrix[int(userId)] - startBeta * itemMatrix[int(itemId)])
                    userMatrix2 = finalAlpha * (wrongGuess * itemMatrix[int(itemId)] - startBeta * userMatrix[int(userId)])
                    # Apply clipping to updates.
                    itemMatrix[int(itemId)] += np.clip(itemMatrix2, -1, 1)
                    userMatrix[int(userId)] += np.clip(userMatrix2, -1, 1)
            # Calculating the mean absolute error (MAE) for the current epoch
            loader.execute('SELECT user_id, item_id, rating FROM train_ratings')
            count = 0
            totalValueError = 0
            for data in dataHandler(loader, dataSize):
                for userId, itemId, rating in data:
                    guess = np.dot(userMatrix[int(userId)], itemMatrix[int(itemId)])
                    totalValueError += abs(rating - guess)
                    count += 1
            # Calculate MAE for the epoch.
            currentMAE = totalValueError / count
            print(f"Epoch {epoch+1}/{epochs} completed - MAE: {currentMAE:.4f}")
            # Early stopping if the improvement in MAE is below the tolerance
            if finalMAE - currentMAE < tolerance:
                print("Early stopping: MAE improvement below tolerance.")
                # Early stopping condition.
                break
            finalMAE = currentMAE

    finally:
        # Saving the user and item matrices to CSV files
        np.savetxt("/Users/Joro/Downloads/COMP3208- Social Computing Techniques/SCT-CW2/predicted_ratingsFinal2.csv", itemMatrix, delimiter=",")
        np.savetxt("/Users/Joro/Downloads/COMP3208- Social Computing Techniques/SCT-CW2/predicted_ratingsFinal1.csv", userMatrix, delimiter=",")
        # Closing the database connection
        dataConnector.close()

    return userMatrix, itemMatrix

def dataLoader(dataFile, trainFile, testFile):
    """
    Loads data from files into SQLite database tables.

    Args:
        dataFile (str): The path to the SQLite database file.
        trainFile (str): The path to the training data file.
        testFile (str): The path to the test data file.

    Returns:
        None
    """
    dataConnector = sqlite3.connect(dataFile)
    loader = dataConnector.cursor()
    # Dropping existing tables if they exist
    loader.execute('DROP TABLE IF EXISTS train_ratings')
    loader.execute('DROP TABLE IF EXISTS test_ratings')
    # Creating new tables for train and test ratings
    loader.execute('''CREATE TABLE train_ratings (user_id INTEGER,item_id INTEGER,rating REAL,timestamp INTEGER)''')
    loader.execute('''CREATE TABLE test_ratings (user_id INTEGER,item_id INTEGER,timestamp INTEGER)''')
    # Loading data from CSV files into the tables
    with open(trainFile, 'r') as doc:
        data = (line.strip().split(',') for line in doc)
        loader.executemany('INSERT INTO train_ratings (user_id, item_id, rating, timestamp) VALUES (?, ?, ?, ?)',data)

    with open(testFile, 'r') as doc:
        data = (line.strip().split(',') for line in doc)
        loader.executemany('INSERT INTO test_ratings (user_id, item_id, timestamp) VALUES (?, ?, ?)',data)
    # Committing the changes to the database
    dataConnector.commit()
    # Closing the database connection
    dataConnector.close()

def predictionLoader(userMatrix, itemMatrix, dataFile, resultFile):
    """
    Loads user and item matrices, retrieves test ratings from a SQLite database,
    performs predictions using dot product of user and item matrices,
    rounds the predictions, and saves the results to a file.

    Args:
        userMatrix (numpy.ndarray): Matrix representing user features.
        itemMatrix (numpy.ndarray): Matrix representing item features.
        dataFile (str): Path to the SQLite database file.
        resultFile (str): Path to the file where the predictions will be saved.

    Returns:
        None
    """
    dataConnector = sqlite3.connect(dataFile)
    loader = dataConnector.cursor()
    guesses = []
    loader.execute('SELECT user_id, item_id, timestamp FROM test_ratings')
    # Performing predictions using the dot product of user and item matrices
    for userId, itemId, timestamp in loader.fetchall():
        firstGuess = np.dot(userMatrix[int(userId)], itemMatrix[int(itemId)])
        guessLimit = np.clip(firstGuess, 0.5, 5.0)
        rounded_prediction = np.round(guessLimit)
        guesses.append(f"{userId},{itemId},{rounded_prediction},{timestamp}\n")
    # Saving the predictions to a file
    with open(resultFile, 'w') as doc:
        doc.writelines(guesses)
    # Closing the database connection
    dataConnector.close()

dataFile = 'recommender_system.db'
trainFile = 'train_20M_withratings.csv'  
testFile = 'test_20M_withoutratings.csv' 
resultFile = 'predicted_ratingsSubmission.csv' 
latentFactors = 50 

# Loading data, factorizing the matrix, and generating predictions
dataLoader(dataFile, trainFile, testFile)
userMatrix, itemMatrix = factoriseMatrix(dataFile, latentFactors)
predictionLoader(userMatrix, itemMatrix, dataFile, testFile, resultFile)
print("Predictions have been saved to the file.")