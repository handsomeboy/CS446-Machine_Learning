"""Input and output helpers to load in data.
"""
import numpy as np

def read_dataset_tf(path_to_dataset_folder,index_filename):
    """ Read dataset into numpy arrays with preprocessing included
    Args:
        path_to_dataset_folder(str): path to the folder containing samples and indexing.txt
        index_filename(str): indexing.txt
    Returns:
        A(numpy.ndarray): sample feature matrix A = [[1, x1], 
                                                     [1, x2], 
                                                     [1, x3],
                                                     .......] 
                                where xi is the 16-dimensional feature of each sample
            
        T(numpy.ndarray): class label vector T = [[y1], 
                                                  [y2], 
                                                  [y3], 
                                                   ...] 
                             where yi is 1/0, the label of each sample 
    """
    ###############################################################
    # Fill your code in this function
    ###############################################################
    # Hint: open(path_to_dataset_folder+'/'+index_filename,'r')
    A = []
    T = []
    with open(path_to_dataset_folder + '/' + index_filename, 'r') as label_file:
        for line in label_file:
            label, filename = line.split()
            T.append(max(int(label),0))
            x = [1]
            with open(path_to_dataset_folder + '/' + filename, 'r') as feature_file:
                num_list = feature_file.readline().split()
                for item in num_list:
                    frac, power = item.split('e')
                    p = int(power[1:])
                    if power[0] == '-': p *= -1
                    x.append(float(frac)*10**p)
            A.append(x)
    A = np.array(A)
    T = np.array(T)
    T = T.reshape(T.shape[0],1)

    return A, T
