#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install keras


# In[2]:


pip install tensorflow


# In[3]:


import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import json
import pickle

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
import random

words=[]
classes = []
documents = []
ignore_words = ['?', '!']
data_file = open('chatbot_data.json').read()
intents = json.loads(data_file)


# In[4]:


for intent in intents['intents']:
    for pattern in intent['patterns']:

        #tokenize each word
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        #add documents in the corpus
        documents.append((w, intent['tag']))

        # add to our classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])


# In[5]:


print(words)


# In[6]:


# lemmatize, lower each word and remove duplicates
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))


# sort classes
classes = sorted(list(set(classes)))
# documents = combination between patterns and intents
print (len(documents), "documents")
# classes = intents
print (len(classes), "classes", classes)
# words = all words, vocabulary
print (len(words), "unique lemmatized words", words)

pickle.dump(words,open('words.pkl','wb'))
pickle.dump(classes,open('classes.pkl','wb'))


# In[7]:


from sklearn.model_selection import train_test_split


# In[8]:


# create our training data
training = []
# create an empty array for our output
output_empty = [0] * len(classes)
# training set, bag of words for each sentence
for doc in documents:
    # initialize our bag of words
    bag = []
    # list of tokenized words for the pattern
    pattern_words = doc[0]
    # lemmatize each word - create base word, in attempt to represent related words
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    # create our bag of words array with 1, if word match found in current pattern
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    # output is a '0' for each tag and '1' for current tag (for each pattern)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])
# shuffle our features and turn into np.array
random.shuffle(training)
training = np.array(training)
# create train and test lists. X - patterns, Y - intents
X = list(training[:,0])
y = list(training[:,1])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, shuffle = False)
print("Training data created")


# In[9]:


print(training)


# In[10]:


print(output_row)


# In[11]:


print(pattern_words)


# In[12]:


print(X)


# In[13]:


print(y)


# X_train = np.array(X_train)
# X_test = np.array(X_test)
# y_train = np.array(y_train)
# y_test = np.array(y_test)
# X = np.array(X)
# y = np.array(y)

# In[14]:


# Create model - 3 layers. First layer 128 neurons, second layer 64 neurons and 3rd output layer contains number of neurons
# equal to number of intents to predict output intent with softmax
model = Sequential()
model.add(Dense(128, input_shape=(len(X_train[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(y_train[0]), activation='softmax'))

# Compile model. Stochastic gradient descent with Nesterov accelerated gradient gives good results for this model
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

#fitting and saving the model 
hist = model.fit(np.array(X_train), np.array(y_train), epochs=200,  batch_size=5, verbose=1)


#ynew = model.predict_classes(np.array(test_x))
#model.save('chatbot_model.h5',hist)

print("model created")


# In[15]:


_, accuracy = model.evaluate(X_test, y_test)
print('Accuracy: %.2f' % (accuracy*100))


# Running the final model to save for GUI

# In[16]:


# Create model - 3 layers. First layer 128 neurons, second layer 64 neurons and 3rd output layer contains number of neurons
# equal to number of intents to predict output intent with softmax
model = Sequential()
model.add(Dense(128, input_shape=(len(X[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(y[0]), activation='softmax'))

# Compile model. Stochastic gradient descent with Nesterov accelerated gradient gives good results for this model
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

#fitting and saving the model 
hist_1 = model.fit(np.array(X), np.array(y), epochs=1000,  batch_size=5, verbose=1)


#ynew = model.predict_classes(np.array(test_x))
model.save('chatbot_model.h5',hist_1)

print("model created")


# saving the same model for chat bot GUI

# In[17]:


import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np

from keras.models import load_model
model = load_model('chatbot_model.h5')
import json
import random
intents = json.loads(open('chatbot_data.json').read())
words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))


# In[18]:


from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
   
ps = PorterStemmer()
#for w in words:
 #   print(w, " : ", ps.stem(w))


# In[19]:


def clean_up_sentence(sentence):
    # tokenize the pattern - split words into array
    sentence_words = nltk.word_tokenize(sentence)
    #sentence_words = [ps.stem(word.lower()) for word in sentence_words]
    # stem each word - create short form for word
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words
# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence

def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words) 
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))

def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words,show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.70
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list


# In[20]:


print(np.array(bag))


# In[21]:


def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
    return result

def chatbot_response(text):
    ints = predict_class(text, model)
    res = getResponse(ints, intents)
    return res


# def getResponse(intents_list, intents_json):
#     tag = intents_list[0]['intent']
#     list_of_intents = intents_json['intents']
#     for i in list_of_intents:
#         if(i['tag']== tag):
#             result = random.choice(i['responses'])
#             break
#     return result
# 
# while True:
#     message = input("")
#     ints=predict_class(message)
#     res = getResponse(ints, intents)
#     print (res)

# In[22]:


import tkinter


# In[23]:


from tkinter import *


# In[24]:


def send():
    msg = EntryBox.get("1.0",'end-1c').strip()
    EntryBox.delete("0.0",END)
    if msg != '':
        ChatLog.config(state=NORMAL)
        ChatLog.insert(END, "You: " + msg + '\n\n')
        ChatLog.config(foreground="white", font=("Verdana", 15, 'bold' ))
        res = chatbot_response(msg)
        ChatLog.insert(END, "Bot: " + res + '\n\n')
        ChatLog.config(state=DISABLED)
        ChatLog.yview(END)


# In[25]:


base = Tk()
base.title("Chatbot")
base.geometry("400x500")
base.resizable(width=TRUE, height=TRUE)


# In[26]:


#Create Chat window
ChatLog = Text(base, bd=0, bg="black", height="10", width="100", font="Verdana",)
ChatLog.config(state=DISABLED)


# In[27]:


#Bind scrollbar to Chat window
scrollbar = Scrollbar(base, command=ChatLog.yview, cursor="mouse")
ChatLog['yscrollcommand'] = scrollbar.set


# In[28]:


#Create Button to send message
SendButton = Button(base, font=("Verdana",12,'bold'), text="Send", width="12", height=5,
                    bd=0, bg="#096484", activebackground="black",fg='white',
                    command= send )


# In[29]:


#Create the box to enter message
EntryBox = Text(base, bd=0, bg="white",width="29", height="5", font="Verdana")
EntryBox.bind("<Return>", send)


# In[30]:


#Place all components on the screen
scrollbar.place(x=376,y=6, height=386)
ChatLog.place(x=6,y=6, height=386, width=370)
EntryBox.place(x=128, y=401, height=90, width=265)
SendButton.place(x=6, y=401, height=90)


# In[ ]:


base.mainloop()


# In[ ]:





# In[ ]:





# In[ ]:




