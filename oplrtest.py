'''
test file of oplr.py
'''
from oplr import run
from oplrer import error_run
from numpy import array

#you need to change input_txt_file_url to your absolute directory below
training_url = r'C:\Users\xieliy\Desktop\master thesis\code\github\training data.txt'
run(training_url)

#test error rate using testing data set and computed classifier
testing_url = r'C:\Users\xieliy\Desktop\master thesis\code\github\testing data.txt'
classifier_url = r'C:\Users\xieliy\Desktop\master thesis\code\github\output classifier.txt'
error_run(testing_url, classifier_url)
