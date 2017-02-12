# Import system
import sys

# Import my functions
from generateDatasets import generateDatasets
from my_cross_val import my_cross_val


def main():
    # Generate Boston50, Boston75 and Digits datasets
    boston50, boston75, digits = generateDatasets()
    
    # Check if correct number of arguments are passsed
    if((len(sys.argv)>1 and len(sys.argv)<4) or len(sys.argv)>4):
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
                return
        else: 
            print("k must an integer")
            return
        
    # Default case: if user is not passing any command line arguments
    else:
        print("No command line arguments passed, running on default datasets\n")
        
        # Fit model and cross validate
        print("Method = LinearSVC   \nDataset = Boston50")
        my_cross_val('LinearSVC', boston50['data'], boston50['class'], 10)
        print("\n")
        
        print("Method = LinearSVC   \nDataset = Boston75")
        my_cross_val('LinearSVC', boston75['data'], boston50['class'], 10)
        print("\n")
        
        print("Method = LinearSVC   \nDataset = Digits")
        my_cross_val('LinearSVC', digits.data, digits.target, 10)
        print("\n")
        
        print("Method = SVC   \nDataset = Boston50")
        my_cross_val('SVC', boston50['data'], boston50['class'], 10)
        print("\n")
        
        print("Method = SVC   \nDataset = Boston75")
        my_cross_val('SVC', boston75['data'], boston75['class'], 10)
        print("\n")
        
        print("Method = SVC   \nDataset = Digits")
        my_cross_val('SVC', digits.data, digits.target, 10)
        print("\n")
        
        print("Method = Logistic Regression   \nDataset = Boston50")
        my_cross_val('LogisticRegression', boston50['data'], boston50['class'], 10)
        print("\n")
        
        print("Method = Logistic Regression   \nDataset = Boston75")
        my_cross_val('LogisticRegression', boston75['data'], boston75['class'], 10)
        print("\n")
        
        print("Method = Logistic Regression   \nDataset = Digits")
        my_cross_val('LogisticRegression', digits.data, digits.target, 10)
        print("\n")
    
if __name__ == '__main__': 
    main()