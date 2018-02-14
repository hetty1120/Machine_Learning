#!/usr/bin/python3
# coding: utf-8

# In[111]:

import numpy as np
import argparse


# In[112]:

def read_data(filepath):
    # read the training data and change it to the data and label
    file_object = open(filepath)
    line = file_object.readline()
    label = []
    train = []
    
    while line:
        
        line = line.strip()
        line = line.split(' ')
        label.append(int(line[0]))
        temp_list = [0] * 123
        for i in range(1,len(line)):
            temp = line[i].find(':')
            feature_num = int(line[i][:temp])
            temp_list[feature_num-1] = 1
        train.append(temp_list)

        line = file_object.readline()
    
    return np.array(train), np.array(label)


# In[113]:

def get_sign(x):
    if x > 0:
        return 1
    elif x == 0:
        return 0
    else:
        return -1


# In[114]:

def acc(data, label, weights, bias):
    
    right_answer = 0
    sample_size = data.shape[0]
    
    for i in range(sample_size):
        result = get_sign(np.dot(weights.transpose(), data[i]) + bias)
        if result == label[i]:
            right_answer += 1
      
    accuracy = right_answer/sample_size
    
    return accuracy


# In[115]:

def training(epochs, c):
    
    sample_size = Train.shape[0]
    w = np.array([0] * Train.shape[1])
    b = 0
    learning_rate=0.1

    for epoch in range(epochs):

        for i in range(sample_size):
            result = np.dot(w.transpose(), Train[i]) + b
            if 1 - Label[i]*result > 0:
                w = w - learning_rate * ( w/sample_size - c * Label[i] * Train[i] ) 
                b = b + learning_rate * c * Label[i]
            else:
                w = w - learning_rate * w/sample_size
    return w,b


# In[121]:

parser = argparse.ArgumentParser(description='SVM')
parser.add_argument('--epochs', type=int, default=5, help='an integer for training iterations')
parser.add_argument('--capacity', default='1', help='a number for the capacity')

args = parser.parse_args()
epochs = args.epochs
c = eval(args.capacity)

Train, Label = read_data('a7a.train')
weights, bias = training(epochs, c)
dev_data, dev_label = read_data('a7a.dev')
test_data, test_label = read_data('a7a.test')
acc_train = acc(Train, Label, weights, bias)
acc_test = acc(test_data, test_label, weights, bias)
acc_dev = acc(dev_data, dev_label, weights, bias)

print('EPOCHS:', epochs)
print('CAPACITY:', c)
print('TRAINING_ACCURACY:', acc_train)
print('TEST_ACCURACY:', acc_test)
print('DEV_ACCURACY:', acc_dev)
parameters = [bias] + weights.tolist()
print('FINAL_SVM:', parameters)


# In[ ]:



