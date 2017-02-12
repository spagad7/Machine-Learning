# Import system
import sys

# Import my function
from generateDatasets import generateDatasets
from my_train_test import my_train_test

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
        
    # Default case: if user is not passing any command line arguments
    else:
        print("No command line arguments passed, running on default datasets\n")
    
        # Fit model, test and report error rates
        print("Method = LinearSVC   \nDataset = Boston50")
        my_train_test('LinearSVC', boston50['data'], boston50['class'], 0.75, 10)
        print("\n")
        
        print("Method = LinearSVC   \nDataset = Boston75")
        my_train_test('LinearSVC', boston75['data'], boston50['class'], 0.75, 10)
        print("\n")
        
        print("Method = LinearSVC   \nDataset = Digits")
        my_train_test('LinearSVC', digits.data, digits.target, 0.75, 10)
        print("\n")
        
        print("Method = SVC   \nDataset = Boston50")
        my_train_test('SVC', boston50['data'], boston50['class'], 0.75, 10)
        print("\n")
        
        print("Method = SVC   \nDataset = Boston75")
        my_train_test('SVC', boston75['data'], boston75['class'], 0.75, 10)
        print("\n")
        
        print("Method = SVC   \nDataset = Digits")
        my_train_test('SVC', digits.data, digits.target, 0.75, 10)
        print("\n")
        
        print("Method = LogisticRegression   \nDataset = Boston50")
        my_train_test('LogisticRegression', boston50['data'], boston50['class'], 0.75, 10)
        print("\n")
        
        print("Method = LogisticRegression   \nDataset = Boston75")
        my_train_test('LogisticRegression', boston75['data'], boston75['class'], 0.75, 10)
        print("\n")
        
        print("Method = LogisticRegression   \nDataset = Digits")
        my_train_test('LogisticRegression', digits.data, digits.target, 0.75, 10)
        print("\n")
# End of main

if __name__ == '__main__': 
    main()