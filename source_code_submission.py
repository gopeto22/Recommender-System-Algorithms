import numpy as np

def load_dataMatrix(entries):
    """
    Loading data from a CSV file
    Returning users, items, their numbers and rankings

    Parameters:
        entries : contains user, item and rankings

    Returns:
        matrixUserItems: Matrix of user-item rankings
    """
    # Extracting unique users and items
    users, items = np.unique(entries[:, 0]), np.unique(entries[:, 1]) 
    
    # Determining the number of unique users and items
    usersNum, itemsNum = len(users), len(items)

    # Declaring a matrix for user item records
    matrixUserItems = np.zeros((usersNum, itemsNum)) 

    # Create a dictionary to match user IDs and items with their indexes
    userIndex = {user: index for index, user in enumerate(users)} 
    itemIndex = {item: index for index, item in enumerate(items)} 

    # Filling the matrix with numbers
    for entry in entries:
        # Get the user and item index
        userInd, itemInd, rankings, timestamp = entry
        userInd = userIndex[userInd]
        itemInd = itemIndex[itemInd]

        # Assign the rankings to the correct user and item index
        matrixUserItems[userInd, itemInd] = rankings

    return matrixUserItems

def cosine_similarity(matrix):

    """
    Cosine similarity between matrix rows with efficient sparse data processing

    Parameters:
        matrix (numpy.ndarray): Matrix of user-object rankings

    Returns:
        similarityMatrix: Matrix of cosine similarity between items
    """

    # Calculation of the dot product between elements
    dotProduct = np.dot(matrix.T, matrix)

    # Calculation of the mean deviation
    meanDeviation = matrix - np.mean(matrix, axis=1, keepdims=True)

    # Number of cols
    length = len(matrix[0])  

    # Calculating the Euclidean/L2 distance for the element
    euclidean = np.linalg.norm(matrix, axis=0)

    # normalisation by using the mean deviation and norm of each column and transposing the matrix
    normalise = meanDeviation / np.linalg.norm(meanDeviation, axis=0).reshape(1,length)  
    normalise = normalise.T  

    # similarity matrix using inner product
    similarityMatrix = np.asarray([[np.inner(normalise[x], normalise[y]) for x in range(length)] for y in range(length)]) 

    # Do not include self-similarity
    np.fill_diagonal(similarityMatrix, 0)

    return similarityMatrix

def train_model(rankingUser, item, similarityMatrix):
    '''
    Training process for cosine similarity algorithm.
    Create a matrix of users and items using the similarity between the items and the actual scores

    Parameters:
        rankingUser : Ranking matrix of users and items
        item : Item index
        similarityMatrix : Matrix of cosine similarity between items

    Returns:
        expectedResult: item ranking prediction
    '''
    # Initialisation of sum containing a weighted ranking and weights
    rankingsWeighted = 0  
    similaritiesWeighted = 0

    # Rated items with a similarity greater than 0
    satisfy_indices = np.where((rankingUser[:, 0] != 0) & (similarityMatrix[:, item] > 0))[0]

    # Sortation of similarity matrix
    totalSimilarity = np.sum(similarityMatrix[satisfy_indices, item] > 0)
    sortedSimilarity = satisfy_indices[np.lexsort((-similarityMatrix[satisfy_indices, item], np.repeat(totalSimilarity, len(satisfy_indices))))] 

    # Top 20 similar items
    topMatch = 16
    sortedSimilarity = sortedSimilarity[:topMatch] 

    # Finding sum of similarity and weighted ranking
    for itemIndex in sortedSimilarity:
        similarity = similarityMatrix[itemIndex, item] 
        ranking = rankingUser[itemIndex, 0] 
        rankingsWeighted += similarity * ranking 
        similaritiesWeighted += similarity 

    # Exclude zero division
    if similaritiesWeighted == 0: 
        return 0

    expectedResult = rankingsWeighted / similaritiesWeighted

    # Rounding to the nearest 0.5
    expectedResult = min(5, max(1, expectedResult)) 

    expectedResult = round(expectedResult) 

    return expectedResult

def predict_rankings(test_data, train_data, similarityMatrix):
    """
    Predicting rankings in test data

    Parameters:
        test_data : Test array with user, item and timestamp
        train_data : Training array with user, item, ranking, and timestamp
        similarityMatrix : Matrix of cosine similarity between items

    Returns:
        predictions: Ranking predictions 
    """
    # Creating a dictionary matching element identifiers with their indexes
    itemIndexList = indexing(train_data)
    predictions = []

    # Predictions for ranking
    for entry in test_data:
        userNum, itemNum, timestamp = entry

        # If the item is not in the training set, provide backup prediction
        if itemNum not in itemIndexList: 
            predictions.append([userNum, itemNum, 2, timestamp]) 
            continue
        
        # Retrieve rankings for items the user has rated
        rankingsUsers = np.zeros((len(itemIndexList), 2))  
        
        entriesUser = train_data[train_data[:, 0] == userNum]  

        # Initialising rankings and timestamps
        for entry1 in entriesUser: 
            itemInd = itemIndexList[entry1[1]]  
            rankingsUsers[itemInd, 0] = entry1[2]  
            rankingsUsers[itemInd, 1] = entry1[3]  

        # Ranking prediction
        predicted_ranking = train_model(rankingsUsers, itemIndexList[itemNum], similarityMatrix)
        predictions.append([int(userNum), int(itemNum), predicted_ranking, timestamp])

    return predictions

def indexing(entries):
    """
    Matching identifiers and indexes

    Parameters:
        entries : contains user, item and rankings

    Returns:
        itemIndex: Matching of item ids abd indices
    """
    # Extracting unique items
    itemInd = np.unique(entries[:, 1]) 

    # Building vocabulary mapping
    itemIndex = {itemId: index for index, itemId in enumerate(itemInd)} 
    
    return itemIndex

def predicionSerialisation(predictions, outputfile):
    """
    Save predictions to a CSV file

    Parameters:
        predictions : Ranking predictions
        outputfile : File returned
    """
    # Saving predictions to a CSV file
    np.savetxt(outputfile, predictions, delimiter=',', fmt='%d,%d,%.1f,%d') 

if __name__ == '__main__':

    # Load data entries
    train_data = np.genfromtxt("train_100k_withratings.csv", delimiter=',', dtype=np.float32)
    test_data = np.genfromtxt("test_100k_withoutratings.csv", delimiter=',', dtype=np.float32)

    # Arrange the data as a matrix and calculate the cosine similarity
    data_matrix = load_dataMatrix(train_data)
    similarity_matrix = cosine_similarity(data_matrix) 

    # Predict ratings
    predictions = predict_rankings(test_data, train_data, similarity_matrix)

    # Save predictions to a CSV file
    predicionSerialisation(predictions, "predictionsSubmission.csv")
