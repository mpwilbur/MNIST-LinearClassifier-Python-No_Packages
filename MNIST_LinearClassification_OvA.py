
import numpy as np
import pandas as pd
import time

def load_train_data(data_file):
    data = pd.read_csv(data_file)
    y_train = data['label'].values
    y_train.shape = (len(y_train), 1)
    
    x_train = data.iloc[0:len(data), 1:]
    x_train = x_train.values
    x_train = x_train/1000
    o = np.ones((np.shape(x_train)[0],1))
    x_train = np.concatenate((o,x_train),axis = 1)
    x_train = x_train.T
    return x_train, y_train

def load_test_data(data_file):
    data = pd.read_csv(data_file)
    x_test = data.values
    x_test = x_test/1000
    o = np.ones((np.shape(x_test)[0],1))
    x_test = np.concatenate((o,x_test),axis = 1)
    x_test = x_test.T
    return x_test
    
def sigmoid(z): 
    y = 1/(1+np.exp(-z))
    return y

def softmax_grad(X, y, wo):
    w = wo
    iter = 1
    grad = 1
    max_its = 2000
    alpha = .0001
    while np.linalg.norm(grad) > 10**(-5) and iter <= max_its:
        #find r vector (note: r = dot(Z, y))
        z_temp = -y*np.dot(X.T, w)
        r = sigmoid(z_temp) * y
        #find gradient
        grad = -np.dot(X, r)
        w -= alpha*grad
        iter += 1
    return w

def y_temp_assign(y_train, val):
    y_temp = [-1 if x != val else 1 for x in y_train]
    y_temp = np.asarray(y_temp)
    y_temp.shape = (len(y_train), 1)
    return y_temp

def OvA_Accuracy(X, Y_real, w_matrix):
    P = len(Y_real)
    # find y_predict
    y_predict = np.dot(X.T, w_matrix)
    y_predict = np.argmax(y_predict, axis = 1)
    y_predict.shape = (P, 1)
    test = y_predict - Y_real
    test = [1 if x != 0 else 0 for x in test]
    accuracy = 1.00 - (sum(test)/P)
    return accuracy

def OvA_Predict(X, w_matrix):
    y_predict = np.dot(X.T, w_matrix)
    y_predict = np.argmax(y_predict, axis = 1)
    y_predict = [x[0] for x in y_predict]
    return y_predict

if __name__ == "__main__":
    # start timer
    start = time.time()
    
    # data file names
    train_data_file = "train.csv"
    test_data_file = "test.csv"
    
    # load data
    x_train, y_train = load_train_data(train_data_file)
    num_features = len(x_train)
    num_class = len(np.unique(y_train))
    
    w_matrix = []
    
    for i in range(num_class):
        y_temp = y_temp_assign(y_train, i)
        wo = np.random.rand(num_features, 1)
        w_temp = softmax_grad(x_train, y_temp, wo)
        w_matrix.append(w_temp)
        
        
    # find accuracy of training set
    accuracy_training = OvA_Accuracy(x_train, y_train, w_matrix)
    s_1 = "The accuracy of the training set is %s"%accuracy_training
    print(s_1)
    
    # get label predictions for testing set
    x_test = load_test_data(test_data_file)
    y_predictions = OvA_Predict(x_test, w_matrix)
    
    # save as file
    imageid = range(len(y_predictions))
    imageid = [x + 1 for x in imageid]
    predictions = pd.DataFrame({'ImageId': imageid,
                                'Label': y_predictions})
    
    # save predictions to a csv file 
    predictions.to_csv('test_predictions.csv', index = False)

end = time.time()
print(end - start)
        
    