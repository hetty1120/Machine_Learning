#!/usr/bin/python3
# coding: utf-8

# In[1]:

import numpy as np
import math
import matplotlib.pyplot as plt


# In[2]:

def load_data(filepath):
    data = np.loadtxt(filepath)
    split_num = int(len(data) * 0.9)
    return np.array(data[:split_num]), np.array(data[split_num:])


# In[3]:

def multi_normal(data, mean, covariance):
    e = (-1/2) * ((data-mean).dot(np.linalg.inv(covariance))).dot((data-mean).T)
    d = math.sqrt((2*math.pi)**covariance.shape[0]*np.linalg.det(covariance))
    result = np.exp(e)/d
    
    return result


# In[4]:

def compute_alpha(data, group, a, mean, covariance):
    
    N = len(data)
    alpha = np.zeros((N+1, group), dtype=np.float128)
    for i in range(group):
        alpha[0][i] = 1
    
    for time in range(1,N+1):
        for i in range(group):
            for j in range(group):
                alpha[time][i] += multi_normal(data[time-1], mean[i], covariance[i]) * a[j][i] * alpha[time-1][j]
    
    return alpha


# In[5]:

def compute_beta(data, group, a, mean, covariance):
    
    N = len(data)
    beta = np.zeros((N+1, group), dtype=np.float128)
    for i in range(group):
        beta[N][i] = 1

    for time in range(N-1,0,-1):
        for i in range(group):
            for j in range(group):
                beta[time][i] += multi_normal(data[time], mean[j], covariance[j]) * a[i][j] * beta[time+1][j]   
    
    return beta


# In[6]:

def hmm(data, dev_data, group, iterations):

    # initialization
    a = np.random.rand(group,group)
    a = [a[i]/sum(a[i]) for i in range(a.shape[0])]
    a = np.array(a, dtype=np.float128)
    
    N = len(data)
    
    #np.random.seed(3)
    random_data = np.random.permutation(data)
    mean = random_data[:group]
    covariance = np.array([np.cov(data.T) for i in range(group)])
    belong = np.zeros((len(data), group), dtype=np.float128)
    
    likelihood = []
    dev_likelihood = []
    
    for epoch in range(iterations):

        # E-step
        alpha = compute_alpha(data, group, a, mean, covariance)
        
        beta = compute_beta(data, group, a, mean, covariance)
        
        for i in range(N):
            for j in range(group):
                belong[i][j] = alpha[i+1][j] * beta[i+1][j]
                
        trans = np.zeros((N,group,group), dtype=np.float128)
        for time in range(N-1):
            for i in range(group):
                for j in range(group):
                    trans[time][i][j] = alpha[time+1][i] * a[i][j] * multi_normal(data[time+1], mean[j], covariance[j]) * beta[time+2][j]
            
        
        # M-step
        # update transition probabilities
        for i in range(group):
            num_sum = 0
            for j in range(group):
                a[i,j] = sum(trans[:,i,j])
                num_sum += a[i,j]
            a[i] /= num_sum
        
        # update mean & covariance
        for k in range(group):
            transpose_belong = np.transpose([belong[:,k]])
            mean[k] = (transpose_belong * data).sum(axis=0) / sum(belong[:,k])
            covariance_total = np.zeros((2,2))
            for n in range(len(data)):
                covariance_total += (belong[n,k]/sum(belong[:,k])) * np.dot(np.transpose([data[n] - mean[k]]), [data[n] - mean[k]])
            covariance[k] = covariance_total
        
        # compute likelihood
        new_alpha = compute_alpha(data, group, a, mean, covariance)
        likelihood.append(np.log(sum(new_alpha[N,:])))
        new_alpha = compute_alpha(dev_data, group, a, mean, covariance)
        dev_likelihood.append(np.log(sum(new_alpha[len(dev_data),:])))

    
    return likelihood, dev_likelihood


# In[11]:

data, dev_data = load_data("points.dat.txt")

iterations = 40

_range = range(1,iterations+1,1)


fig, ax = plt.subplots(1,2,figsize=(20,8))

for group in range(7):
    likelihood, likelihood_dev = hmm(data, dev_data, group+2, iterations)
    ax[0].plot(_range, likelihood, label = "{} hidden states".format(group+2))
    ax[1].plot(_range, likelihood_dev, label = "{} hidden states".format(group+2))
       

for i in range(2):
    ax[i].set_xlim([1,iterations+1])
    ax[i].set_xlabel('Iterations')
    ax[i].set_ylabel('Log Likelihood')
    ax[i].legend(loc="lower right",fontsize="small")

ax[0].set_title('Training Data')
ax[1].set_title('Development Data')

fig.tight_layout()
plt.show()


# In[1]:

fig.savefig('plot.png')
plt.close(fig)


# In[ ]:



