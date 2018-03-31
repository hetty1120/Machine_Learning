
# coding: utf-8

# In[55]:

import numpy as np
import math
import matplotlib.pyplot as plt


# In[56]:

def load_data(filepath):
    data = np.loadtxt(filepath)
    split_num = int(len(data) * 0.9)
    return np.array(data[:split_num]), np.array(data[split_num:])


# In[57]:

def log_likelihood(data, pai, mean, covariance):
    log_sum = 0
    for n in range(len(data)):
        total = 0
        for k in range(len(pai)):
            total += pai[k] * multi_normal(data[n], mean[k], covariance[k])
        log_sum += np.log(total)  
        
    return log_sum


# In[58]:

def multi_normal(data, mean, covariance):
    e = (-1/2) * ((data-mean).dot(np.linalg.inv(covariance))).dot((data-mean).T)
    d = math.sqrt((2*math.pi)**covariance.shape[0]*np.linalg.det(covariance))
    result = np.exp(e)/d
    
    return result


# In[ ]:

def gmm(data, dev_data, group, iterations, same=False):

    # initialization
    np.random.seed(3)
    random_data = np.random.permutation(data)
    mean = random_data[:group]
    covariance = np.array([np.cov(data.T) for i in range(group)])
    belong = np.zeros((len(data), group))
    pai = np.array([1/group] * group)
    N = len(data)
    
    likelihood = []
    likelihood_dev = []
    
    for i in range(iterations):
        
        # expectation
        for n in range(len(data)):
            normalizaiton = 0
            for k in range(group):
                belong[n,k] = pai[k] * multi_normal(data[n], mean[k], covariance[k])
                normalizaiton += belong[n,k]
            belong[n] /= normalizaiton
        
        # maximizaiton
        for k in range(group):
            pai[k] = sum(belong[:,k]) / N
            tranpose_belong = np.transpose([belong[:,k]])
            mean[k] = (tranpose_belong * data).sum(axis=0) / sum(belong[:,k])
            covariance_total = np.zeros((2,2))
            for n in range(len(data)):
                covariance_total += belong[n,k] * np.dot(np.transpose([data[n] - mean[k]]), [data[n] - mean[k]])
            covariance[k] = covariance_total / sum(belong[:,k]) 
        
        if same:
            covariance_total = np.zeros((2,2))
            for k in range(group):
                covariance_total += pai[k] * covariance[k]
            for k in range(group):
                covariance[k] = covariance_total
            
        likelihood.append(log_likelihood(data, pai, mean, covariance))
        likelihood_dev.append(log_likelihood(dev_data, pai, mean, covariance))
    
    return likelihood, likelihood_dev


# In[ ]:

data, dev_data = load_data("points.dat.txt")

iterations = 60

_range = range(1,iterations+1,1)

fig, ax = plt.subplots(2,2,figsize=(13,8))

for group in range(5):
    likelihood, likelihood_dev = gmm(data, dev_data, group+2, iterations)
    likelihood_tied, likelihood_dev_tied = gmm(data, dev_data, group+2, iterations, True)
    ax[0][0].plot(_range, likelihood, label = "{} clusters".format(group+2))
    ax[0][1].plot(_range, likelihood_dev, label = "{} clusters".format(group+2))
    ax[1][0].plot(_range, likelihood_tied, label = "{} clusters".format(group+2))
    ax[1][1].plot(_range, likelihood_dev_tied, label = "{} clusters".format(group+2))    

for i in range(2):
    for j in range(2):
        ax[i][j].set_xlim([1,iterations+1])
        ax[i][j].set_xlabel('Iterations')
        ax[i][j].set_ylabel('Log Likelihood')
        ax[i][j].legend(loc="lower right",fontsize="small")

ax[0][0].set_title('Training Data with Seperate Covariance')
ax[0][1].set_title('Development Data with Seperate Covariance')
ax[1][0].set_title('Training Data with Tied Covariance')
ax[1][1].set_title('Development Data with Tied Covariance')

fig.tight_layout()
plt.show()


# In[ ]:

fig.savefig('plot.png')
plt.close(fig)


# In[ ]:



