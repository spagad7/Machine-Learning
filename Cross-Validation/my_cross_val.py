# Import System
import sys

# Import numpy
import numpy as np

# Import sklearn
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import cross_val_score

# Import models from scikit-learn
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

# Import my functions
from generateDatasets import generateDatasets



def my_cross_val(method, X, y, k):
    # Divide the dataset into k subsets(list of lists)
    # Run for loop to train on k-1 subsets and test on 1 subset
    # Cycle through the subsets and repeat the process k times
    
    # Init
    dataSubsets = []
    targetSubsets = []
    accuracyScore = []
    errorRate = []
    dataSetSize = len(X)
    dataSubsetSize = int(dataSetSize/k)
    start = 0
    end = start + dataSubsetSize
    
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
    
    # Generate k datasubsets
    for i in range(k):
        #print("Start=", start, "End=", end-1)
        dataSubsets.append(X[start:end])
        targetSubsets.append(y[start:end])
        
        start = end
        if(i==k-2):
            end = dataSetSize
        else:
            end = end + dataSubsetSize
            
    # Perform k-fold train test cross validation         
    indicies = list(range(k))
    
    for i in range(k):
        trainData = list()
        trainTarget = list()
        
        # Generate training set for index from 0 to k-2
        for index in indicies[:-1]:
            trainData.append(dataSubsets[index])
            trainTarget.append(targetSubsets[index])
        
        # Flatten 3D data list to 2d data list
        flatTrainData = np.array([list(item) for subData in trainData for item in subData])
        
        # Flatten 2D target list to 1d target list
        flatTrainTarget = np.array([item for subTarget in trainTarget for item in subTarget])
        
        # Generate test set from k-1th index
        testData = dataSubsets[indicies[-1]]
        testTarget = targetSubsets[indicies[-1]]
        
        # Fit model
        model.fit(flatTrainData, flatTrainTarget)
        # Test model
        result = model.predict(testData)
        
        # Set error counter to 0
        errorCount = 0
        
        # Calculate Error Rate
        for i in range(len(testTarget)):
            if result[i] != testTarget[i]:
                errorCount+=1
        
        # Calculate and save error rate
        errorRate.append((errorCount/len(testTarget))*100)
        
        # Update indicies list
        indicies = indicies[1:] + [indicies[0]]
    
    # Print Results
    print("Error = ", errorRate)
    print("Average Error = ", np.mean(errorRate))
    print("Standard Deviation of Error = ", np.std(errorRate))
    
# End of myCrossVal



def main():
    # Generate Boston50, Boston75 and Digits datasets
    boston50, boston75, digits = generateDatasets()
    
    # Check if correct number of arguments are passsed
    if(len(sys.argv)>1 and len(sys.argv)<4):
        print("Insufficient number of arguments") 
        print("Pass 3 arguments: modelname[LinearSVC, SVC, LogisticRegression], dataset-name[Boston50 | Boston75 | Digits], k")
        return
    # If user is passing command line arguments
    elif(len(sys.argv) == 4):
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
    
        # Check if k is not digit
        if(sys.argv[3].isdigit()):
            if(int(sys.argv[3]) > 0):
                # Call my_cross_val
                my_cross_val(model, dataSet['data'], dataSet['class'], int(sys.argv[3]))
            else:
                print("k must be a positive integer")
        else: 
            print("k must an integer")
    else:
        print("Insufficient number of arguments") 
        print("Pass 3 arguments: modelname[LinearSVC, SVC, LogisticRegression], dataset-name[Boston50 | Boston75 | Digits], k")
    
if __name__ == '__main__':
    main()