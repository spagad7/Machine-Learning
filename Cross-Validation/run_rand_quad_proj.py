# Import System
import sys

# Import sklearn 
from sklearn.datasets import load_digits

# Import my files
from rand_proj import rand_proj
from quad_proj import quad_proj
from my_cross_val import my_cross_val
from generateDatasets import generateDatasets


def main():
        
    # Check if correct number of arguments are passsed
    if((len(sys.argv)>1 and len(sys.argv)<5) or len(sys.argv)>5):
        print("Insufficient number of arguments") 
        print("Pass 4 arguments: modelname[LinearSVC, SVC, LogisticRegression], dataset-name[Boston50 | Boston75 | Digits], projection[rand_proj | quad_proj], k")
        return
        
    # If user is passing command line arguments
    elif(len(sys.argv) == 5):
        # Generate Boston50, Boston75 and Digits datasets
        boston50, boston75, digits = generateDatasets()
        
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
        if(sys.argv[4].isdigit()):
            if(int(sys.argv[4]) > 1):
                k = int(sys.argv[4])
            else:
                print("k must be a positive integer greater than 1")
                return
        else: 
            print("k must an integer")
            return
            
        # Set Projection Type
        if(sys.argv[3] == 'rand_proj'):
            # Call rand_proj
            X1 = rand_proj(dataSet['data'],32)
            my_cross_val(model, X1, dataSet['class'], k)
            
        elif(sys.argv[3] == 'quad_proj'):
            # Call quad_proj
            X2 = quad_proj(dataSet['data'])
            my_cross_val(model, X2, dataSet['class'], k)
        else:
            print("Unsupported projection type!")
            return
        
    # Default case: if user is not passing any command line arguments
    else:
        print("No command line arguments passed, running on default datasets\n")

        # Load digits data
        digits = load_digits()
        
        
        # Q. 4.1
        print("Q. 4.1\n")
        
        # Call rand_proj
        X1 = rand_proj(digits.data,32)
        
        # Perform cross validation
        print("Method = LinearSVC   \nDataSet = Digits    \nk = 10")
        my_cross_val('LinearSVC', X1, digits.target, 10)
        print("\n")
        
        print("Method = SVC   \nDataSet = Digits    \nk = 10")
        my_cross_val('SVC', X1, digits.target, 10)
        print("\n")
        
        print("Method = LogisticRegression   \nDataSet = Digits    \nk = 10")
        my_cross_val('LogisticRegression', X1, digits.target, 10)
        print("\n")
        
        
        # Q. 4.2
        print("Q. 4.2\n")
        
        # Call quad_proj
        X2 = quad_proj(digits.data)
        
        # Perform cross validation
        print("Method = LinearSVC   \nDataSet = Digits    \nk = 10")
        my_cross_val('LinearSVC', X2, digits.target, 10)
        print("\n")
        
        print("Method = SVC   \nDataSet = Digits    \nk = 10")
        my_cross_val('SVC', X2, digits.target, 10)
        print("\n")
        
        print("Method = LogisticRegression   \nDataSet = Digits    \nk = 10")
        my_cross_val('LogisticRegression', X2, digits.target, 10)
        print("\n")
        
        
        # Q. 4.3
        print("Q. 4.3\n")
        
        # Call rand_proj on X2
        X3 = rand_proj(X2, 64)
        
        # Perform cross validation
        print("Method = LinearSVC   \nDataSet = Digits    \nk = 10")
        my_cross_val('LinearSVC', X3, digits.target, 10)
        print("\n")
        
        print("Method = SVC   \nDataSet = Digits    \nk = 10")
        my_cross_val('SVC', X3, digits.target, 10)
        print("\n")
        
        print("Method = LogisticRegression   \nDataSet = Digits    \nk = 10")
        my_cross_val('LogisticRegression', X3, digits.target, 10)
        print("\n")
        
# End of main
    
if __name__ == '__main__':
    main()