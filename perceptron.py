#!/usr/bin/python3

# coding: utf-8

# In[200]:

import numpy as np
import argparse


# In[96]:

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
        temp_list = [0] * 124
        for i in range(1,len(line)):
            temp = line[i].find(':')
            feature_num = int(line[i][:temp])
            temp_list[feature_num-1] = 1
        temp_list[123] = 1
        train.append(temp_list)

        line = file_object.readline()
    
    return np.array(train), np.array(label)


# In[97]:

def get_sign(x):
    if x > 0:
        return 1
    elif x == 0:
        return 0
    else:
        return -1


# In[150]:

def acc(data, label, weights):
    
    right_answer = 0
    sample_size = data.shape[0]
    
    for i in range(sample_size):
        result = get_sign(np.dot(weights.transpose(), data[i]))
        if result == label[i]:
            right_answer += 1
      
    accuracy = right_answer/sample_size
    
    return accuracy


# In[181]:

def training(Train, Label, epochs=15):

    sample_size = Train.shape[0]
    w = np.array([0] * Train.shape[1])

    for epoch in range(epochs):   
        for i in range(sample_size):
            result = get_sign(np.dot(w.transpose(), Train[i]))
            if result != Label[i]:
                w = w + Label[i] * Train[i]
    
    return w


# In[201]:

parser = argparse.ArgumentParser(description='perceptron')
parser.add_argument('--iterations', type=int, default=30)

args = parser.parse_args()
epochs = args.iterations

Train, Label = read_data('a7a.train')
weights = training(Train, Label, epochs)
test_data, test_label = read_data('a7a.test')
acc_test = acc(test_data, test_label,weights)


# In[196]:

print('Test accuracy: %f'%acc_test)

# In[199]:

weights_output = ' '.join([repr(float(k)) for k in weights.tolist()])
print('Feature weights (bias last):', end=' ')
print(weights_output)


# In[ ]:



