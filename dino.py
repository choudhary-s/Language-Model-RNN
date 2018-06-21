import numpy as np
import random
from utils import *

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x),axis=0)

def clip(gradients, maxValue):
    dWaa, dWax, dWya, db, dby = gradients["dWaa"], gradients["dWax"], gradients["dWya"], gradients["db"], gradients["dby"]
    for gradient in [dWaa, dWax, dWya, db, dby]:
        np.clip(gradient, -maxValue, maxValue, out=gradient)
    gradients = {"dWaa":dWaa, "dWax":dWax, "dWya":dWya, "db":db, "dby":dby}
    return gradients

def sample(parameters, char_to_int):
    Waa = parameters["Waa"]
    Wax = parameters["Wax"]
    Wya = parameters["Wya"]
    by = parameters["by"]
    b = parameters["b"]
    
    vocab_size = by.shape[0]
    na = Waa.shape[0]
    
    x = np.zeros((vocab_size,1))
    a = np.zeros((na,1))
    
    indices = []
    
    idx = -1
    counter = 0
    
    while(idx!=char_to_int['\n'] and counter!= 50):
        a = np.tanh(np.dot(Wax, x)+np.dot(Waa, a)+b)
        y = softmax(np.dot(Wya, a)+by)
        
        idx = np.random.choice(list(range(vocab_size)), p=y.ravel())
        indices.append(idx)
        
        x = np.zeros((vocab_size, 1))
        x[idx] = 1
        
        counter += 1
    if(counter==50):
        indices.append(char_to_int['\n'])
    return indices

def optimize(X, Y, a_prev, parameters, learning_rate):
    
    loss, cache = rnn_forward(X, Y, a_prev, parameters, vocab_size)
    gradients, a = rnn_backward(X, Y, parameters, cache)
    gradients = clip(gradients,5)
    parameters = update_parameters(parameters, gradients, learning_rate)
    
    return loss, gradients, a[len(X)-1]

def print_sample(sampled_indices, int_to_char):
    str = ""
    for j in sampled_indices:
        str += int_to_char[j]
    print(str)

def model(data, int_to_char, char_to_int, num_iterations, na, dino_names, vocab_size):
    nx, ny = vocab_size, vocab_size
    parameters = initialize_parameters(na, nx, ny)
    loss = get_initial_loss(vocab_size, dino_names)
    
    with open("names.txt") as f:
        examples = f.readlines()
        examples = [x.lower().strip() for x in examples]
    np.random.seed(0)
    np.random.shuffle(examples)
    
    a_prev = np.zeros((na, 1))
    
    for j in range(num_iterations):
        index = j%len(examples)
        #print("example at index "+str(index)+" is "+str(examples[index]))
        X = [None] + [char_to_int[ch] for ch in examples[index]]
        #print(X)
        Y = X[1:] + [char_to_int["\n"]]
        #print(Y)
        
        curr_loss, gradients, a_prev = optimize(X, Y, a_prev, parameters, learning_rate=0.001)
        loss = smooth(loss, curr_loss)
        
        if j%2000 == 0:
            print("iteration %d loss %f\n"%(j, loss))
            for name in range(dino_names):
                sampled_indices = sample(parameters, char_to_int)
                print_sample(sampled_indices, int_to_char)
                #print("\n")
                
    return parameters

data = open('names.txt', 'r').read()
data = data.lower()
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
print("There are %d total characters and %d unique characters in your data"%(data_size, vocab_size))

char_to_int = {ch:i for i,ch in enumerate(sorted(chars))}
int_to_char = {i:ch for i,ch in enumerate(sorted(chars))}
print(int_to_char)

parameters = model(data, int_to_char, char_to_int, 300000, 50, 7, vocab_size)
