class KNN:
    
    """ For the titanic dataset, the best prediction for k is seen as 3.
    Keeping default value of k as 3. To enable calculation with all distances, p is also taken.
    For this dataset, we will use p=2 as default which becomes Euclidean distance. """
    
    """Value of p can be changed to use Minkowski Distance. """
    
    def __init__(self, k=3, p=2):
        self.k = k
        self.p = p

    def fit(self, X_train, y_train):
        """ This function is initialized to take input of training features and training labels"""
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        
        """ This function is initialized to predict the values for test dataset of titanic."""
        
        minkowski_matrix = self.calculate_minkowski_distance(X_test)
        return self.final_classified_output(minkowski_matrix)

    def calculate_minkowski_distance(self, X_test):
        
        """ Formula of the Minkowski distance is kept in this function to calculate distances 
        between trained and test data."""
        
        col_test_length, col_test_width = X_test.shape
        
        col_train_length, col_train_width = self.X_train.shape
        
        # First create a matrix with number of test rows as rows and number of train rows as columns  
        
        minkowski_matrix = np.zeros((col_test_length, col_train_length))

        # Calculate distance for each row value in test data with each value in training data.
        for i in range(col_test_length):
            for j in range(col_train_length):
                # Distance calculation
                minkowski_matrix[i, j] = (np.sum(abs(X_test[i, :] - self.X_train[j, :]) ** self.p))**(1/self.p)
        return minkowski_matrix

    def final_classified_output(self, minkowski_matrix):
        #print(minkowski_matrix) #Debugging
        col_test_length, col_test_width = minkowski_matrix.shape
        classify_array = np.zeros(col_test_length)

        for i in range(col_test_length):
            # Finding the closest points indexes for ith array.
            index_sorting = np.argsort(minkowski_matrix[i, :])
            
            # From the sorted array, fetch the value for ith array's k closest points.
            nearbyclasses = np.array(self.y_train)[index_sorting[:self.k]]
            
            # Find class which is seen highest in the k nearest values and then fetch it using argmax. 
            classify_array[i] = np.argmax(np.bincount(nearbyclasses))

        return classify_array
