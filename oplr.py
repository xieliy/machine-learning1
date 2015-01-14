'''
Welcome to use our differentially private machine learning algorithm!
Please see 'User Guide' file in our github account for all the details!
Thank you and we welcome any kind advices and opinions!
'''
##implement differentially private empirical risk minimization paper
##output perturbation with logistic regression loss

from numpy import *
from pylab import norm
from scipy.optimize import minimize
from sklearn.cross_validation import train_test_split
from random import SystemRandom

rng = SystemRandom()#generate random numbers from sources provided by the operating system.
seed = rng.seed()#initialize the basic random number generator use current system time
Epsilon_global = 0
Lambda_global = 0

def load_private_dataset(txt_file_url):
    '''
    load private data from .txt file
    '''
    data_set = []#store data set
    txt_file_object = open(txt_file_url, 'r')
    for line in txt_file_object:
        data_set.append(line)
    txt_file_object.close()
    return data_set

def stringToInt(lists):
    '''
    Transform components in every data point from string to int
    '''
    lists2 = []
    for i in range(len(lists)):
        List = []#store every row
        for j in range(len(lists[0])):
            List.append(int(lists[i][j]))
        lists2.append(List)
    return lists2

def separation(strings):
    '''
    Remove character '\n' except last line
    Transform data points from string type to list type
    Transform components in every data point from string to int
    '''
    lists = []
    for i in range(len(strings)-1):
        lists.append(strings[i][:-1].split(' '))
    lists.append(strings[len(strings)-1].split(' '))
    lists = stringToInt(lists)
    return lists#every row is a data point, the last element is the corresponding label

def data_label_split(datalabels):
    '''
    split data part and label part.
    '''
    l = len(datalabels[0])
    data = datalabels[:,0:(l-1)]
    labels = datalabels[:,l-1]
    return (data,labels)

def parameters_set(Epsilon_set,Lambda_set):
    '''
    set parameters used in this paper
    '''
    global Epsilon_global, Lambda_global
    Epsilon_global = Epsilon_set#privacy parameter
    Lambda_global = Lambda_set#regularization parameter
    return

def parameters():
    '''
    set parameters used in this paper
    '''
    Epsilon = Epsilon_global#privacy parameter
    Lambda = Lambda_global#regularization parameter
    return (Epsilon, Lambda)

def noisevector(scale, Length):
    '''
    Generate noise vector with Length
    Generate a vector according to uniform distribution with norm 1, length numDim
    '''
    r1 = random.normal(0, 1, Length)#standard normal distribution
    n1 = norm(r1)#get the norm of this random vector
    r2 = r1/n1#the norm of r2 is 1
    #norm(r2)#this should be 1
    normn = random.gamma(Length, 1/scale, 1)#Generate the norm of noise according to gamma distribution
    res = r2*normn#get the result noise vector
    return res

def lr(z):
    '''
    logistic regression loss function
    '''
    logr = log(1+exp(-z))
    return logr

def lr_output_train(data, labels):
    '''
    output peturbation
    logistic regression
    '''
    L = len(labels)
    l = len(data[0])#length of a data point
    scale = L*parameters()[1]*parameters()[0]/2#chaudhuri2011differentially corollary 11, part 1
    noise = noisevector(scale, l)
    x0 = zeros(l)#starting point with same length as any data point

    print('start training!')
    def obj_func(x):
        jfd = lr(labels[0]*dot(data[0],x))
        for i in range(1,L):
            jfd = jfd + lr(labels[i]*dot(data[i],x))
        f = (1/L)*jfd + (1/2)*parameters()[1]*(norm(x)**2)
        return f
    
    #minimization procedure
    f = minimize(obj_func,x0,method='Nelder-Mead').x#empirical risk minimization using scipy.optimize minimize function
    fpriv = f + noise
    print('training end!')
    return fpriv

def train(data, labels):
    '''
    train output peturbation and output classifer
    ''' 
    classifer_output = lr_output_train(data, labels)
    return classifer_output

def change_file(input_txt_file_url):
    '''
    Since we need to output 'output.txt' file in the same directory with
    input file, we need to change the URL string 
    '''
    for i in range(len(input_txt_file_url)):
        if input_txt_file_url[len(input_txt_file_url)-i-1] == '\\':
            input_txt_file_url = input_txt_file_url[:len(input_txt_file_url)-i]
            ourput_txt_file_url = input_txt_file_url + 'output classifier.txt'
            break
    return ourput_txt_file_url

def write_txt(output,ourput_txt_file_url):
    '''
    write the output into 'output.txt' file
    '''
    with open(ourput_txt_file_url, 'w') as f_out:
        for i in output:
            if i == len(output)-1:
                f_out.write(str(i))
            f_out.write(str(i) + ' ')
    return

def run(input_txt_file_url):
    '''
    The only function that others can call
    '''
    print(__doc__)
    
    print('please set the parameters before continue.')
    Epsilon_set = input('please enter privacy parameter epsilon: ')
    print ('Epsilon is: ' + Epsilon_set)
    Lambda_set = input('please enter regularization parameter lambda: ')
    print ('lambda is: ' + Lambda_set)
    parameters_set(float(Epsilon_set),float(Lambda_set))#input are string, convert to float
    
    data_set = load_private_dataset(input_txt_file_url)
    print('training data load successfully!')
    
    data_set_n = separation(data_set)
    data_set_n = array(data_set_n)
    data,labels = data_label_split(data_set_n)
    print('labels are: ' + str(labels))
    
    classifer_output = train(data, labels)
    print('compute classifer successfully!')
    print('classifer_output is: ' + str(classifer_output))
    
    ourput_txt_file_url = change_file(input_txt_file_url)
    write_txt(classifer_output,ourput_txt_file_url)
    print('write classifer successfully!')
    
    
