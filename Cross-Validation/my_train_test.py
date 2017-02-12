# Import System
import sys

# Import numpy
import numpy as np

# Import sci-kit learn
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import cross_val_score

# Import models from scikit-learn
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

# Import my functions
from generateDatasets import generateDatasets



# Fucntion to split input dataset into train and test set
# based on the value of pi, and calculate error of prediction
def my_train_test(method, X, y, pi, k):
    # Init
    trainData = []
    trainTarget = []
    testData = []
    testTarget = []
    errorRate = []
    dataSetSize = len(X)
    
    # Init model
    if(method == 'LinearSVC'):
        model = LinearSVC()
    elif(method == 'SVC'):
        model = SVC()
    elif(method == 'LogisticRegression'):
        model = LogisticRegression(penalty='l2')
    else:
        print("Method not supported!")
        return
    
    # Repeat the process k times
    for i in range(k):
        # Split Dataset into Train and Test
        trainData = X[:int(dataSetSize*pi)]
        trainTarget = y[:int(dataSetSize*pi)]
        testData = X[int(dataSetSize*pi):]
        testTarget = y[int(dataSetSize*pi):]
        
        # Fit model on training data
        model.fit(trainData, trainTarget)
        # Test model on test data
        result = model.predict(testData)
        
        # Set error counter to 0
        errorCount = 0
        
        # Calculate Error Rate
        for i in range(len(testTarget)):
            if result[i] != testTarget[i]:
                errorCount+=1
        
        # Calculate and save error rate
        errorRate.append((errorCount/len(testTarget))*100)
    
    # Print Results
    print("Error = ", errorRate)
    print("Average Error = ", np.mean(errorRate))
    print("Standard Deviation of Error = ", np.std(errorRate))

# End of my_train_test



# Main function executes only if this file is executed
def main():
    # Generate Boston50, Boston75 and Digits datasets
    boston50, boston75, digits = generateDatasets()
    
    # Check if correct number of arguments are passsed
    if((len(sys.argv)>1 and len(sys.argv)<5) or len(sys.argv)>5):
        print("Insufficient number of arguments") 
        print("Pass 4 arguments: modelname[LinearSVC, SVC, LogisticRegression], dataset-name[Boston50 | Boston75 | Digits], pi, k")
        return
        
    # If user is passing command line arguments
    elif(len(sys.argv) == 5):
        # Set Model
        model = sys.argv[1]
        
        # Set Dataset
        if(sys.argv[2] == 'Boston50'):
            dataSet = boston50
        elif(sys.argv[2] == 'Boston75'):
            dataSet = boston75
        elif(sys.argv[2] == 'Digits'):
            dataSet = digits
        else:
            print("DataSet not supported!")
            return
        
        # Check if value of pi is between 0 and 1
        if(float(sys.argv[3]) <= 0 or float(sys.argv[3]) >= 1):
            print("Value of pi should be between 0 and 1, excluding")
            return
            
        # Check if k is not digit
        if(sys.argv[4].isdigit()):
            if(int(sys.argv[4]) > 0):
                # Call my_cross_val
                my_train_test(model, dataSet['data'], dataSet['class'], float(sys.argv[3]), int(sys.argv[4]))
            else:
                print("k must be a positive integer")
                return
        else: 
            print("k must an integer")
            return
        
    else:
       print("Insufficient number of arguments") 
       print("Pass 4 arguments: modelname[LinearSVC, SVC, LogisticRegression], dataset-name[Boston50 | Boston75 | Digits], pi, k")
    
if __name__ == '__main__':
    main()