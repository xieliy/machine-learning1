'''
Compute error rate, 
'''
from numpy import array, dot

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

def load_classifier(classifier_url):
    '''
    load classifier from .txt file
    '''
    classifier_object = open(classifier_url, 'r')
    classifier = classifier_object.readlines()
    classifier_object.close()
    return classifier

def handle_classifier(classifier):
    '''
    Remove blank ' ' in the last
    Separate whole string into strings 
    Transform components from string to float
    '''
    classifier = classifier[0][:-1].split(' ')
    List = []
    for i in range(len(classifier)):
            List.append(float(classifier[i]))
    return List
    
def error_rate_compute(data, labels, classifier):
    '''
    Compute error rate using classifer on testing data set
    '''
    e = 0#number of errors
    n = len(labels)
    for i in range(n):
        sign = labels[i]*dot(classifier,data[i])
        if sign<0:
            e = e + 1
    res = e/n
    return res

def error_run(testing_url, classifier_url):
    '''
    The only function that others can call
    '''
    data_set = load_private_dataset(testing_url)
    print('testing data load successfully!')
    data_set_n = separation(data_set)
    data_set_n = array(data_set_n)
    data,labels = data_label_split(data_set_n)

    classifier = load_classifier(classifier_url)
    print('classifier load successfully!')
    print('classifier is: ' + str(classifier))
    
    classifier = handle_classifier(classifier)
    print('classifier is: ' + str(classifier))
    
    error_rate = error_rate_compute(data, labels, classifier)
    print(error_rate)
    
