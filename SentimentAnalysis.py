# for reading the file and processing the data in the form of data frame
import pandas as pd 
# for processing textual data
from textblob import TextBlob
# for visualization
import matplotlib.pyplot as plt
# mathematical processing
import numpy as np
import math
# data transformation 
from sklearn.feature_extraction.text import CountVectorizer 

#for preprocessing text information
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

# for model training
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# reading the file using pandas
data = pd.read_pickle('corpus.pkl')

# extracting values for dependent and independent attributes
pol = lambda x: TextBlob(x).sentiment.polarity # is the wor negative or positive
sub = lambda x: TextBlob(x).sentiment.subjectivity # is it a fact or an subjective opinion

# applying polarity and subjectivity of specific record
data['polarity'] = data['transcript'].apply(pol)
data['subjectivity'] = data['transcript'].apply(sub)


# -------------------------  visualizing the data  ------------------------------
# visualizing for every artist
plt.rcParams['figure.figsize'] = [10, 8]
for index, comedian in enumerate(data.index):
    x = data.polarity.loc[comedian]
    y = data.subjectivity.loc[comedian]
    plt.scatter(x, y, color='blue')
    plt.text(x+.001, y+.001, data['full_name'][index], fontsize=10)
    plt.xlim(-.01, .12) 
    
plt.title('Sentiment Analysis', fontsize=20)
plt.xlabel('<-- Negative -------- Positive -->', fontsize=15)
plt.ylabel('<-- Facts -------- Opinions -->', fontsize=15)
plt.show()

# ---------------------------  preprocessing data  -------------------------------
def split_text(text, n=10):
    '''Takes in a string of text and splits into n equal parts, with a default of 10 equal parts.'''

    # Calculate length of text, the size of each chunk of text and the starting points of each chunk of text
    length = len(text)
    size = math.floor(length / n)
    start = np.arange(0, length, size)
    
    # Pull out equally sized pieces of text and put it into a list
    split_list = []
    for piece in range(n):
        split_list.append(text[start[piece]:start[piece]+size])
    return split_list

# list to hold all of the pieces of text
list_pieces = []
for t in data.transcript:
    split = split_text(t)
    list_pieces.append(split)

# Calculating the polarity for each piece of text
polarity_transcript = []
for lp in list_pieces:
    polarity_piece = []
    for p in lp:
        polarity_piece.append(TextBlob(p).sentiment.polarity)
    polarity_transcript.append(polarity_piece)

    
#  ------------------------------- visualizing data  ------------------------------
# visualizing for individual artist
plt.rcParams['figure.figsize'] = [16, 12]
for index, comedian in enumerate(data.index):    
    plt.subplot(3, 4, index+1)
    plt.plot(polarity_transcript[index])
    plt.plot(np.arange(0,10), np.zeros(10))
    plt.title(data['full_name'][index])
    plt.ylim(ymin=-.2, ymax=.3)
plt.suptitle('polarity analysis', fontsize=20)
plt.show()


# list to hold all of text subtext's subjectivity
sub_transcript = []
for lp in list_pieces:
    sub_piece = []
    for p in lp:
        sub_piece.append(TextBlob(p).sentiment.subjectivity)
    sub_transcript.append(sub_piece)

#  ----------------------------- visualizing data ------------------------------------
# subjectivity for every artist indiviually
plt.rcParams['figure.figsize'] = [16, 12]
for index, comedian in enumerate(data.index):    
    plt.subplot(3, 4, index+1)
    plt.plot(sub_transcript[index])
    plt.plot(np.arange(0,10), np.zeros(10))
    plt.title(data['full_name'][index])
    plt.ylim(ymin=-.2, ymax=.7)
plt.suptitle('subjectivity Analysis', fontsize=20)
plt.show()


# ------------------------------- model training  -------------------------------
# for model training
texts=[]
for i in range(len(data['transcript'])):
    texts.append(data['transcript'][i])


#--------------------------------------  data preprocessing  -------------------------

words = set(nltk.corpus.words.words())

i=0
for example_sent in texts:

    stop_words = set(stopwords.words('english'))

    word_tokens = word_tokenize(example_sent)
    # converts the words in word_tokens to lower case and then checks whether they are present in stop_words or not
    filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words]
    #with no lower case conversion
    filtered_sentence = []

    for w in word_tokens:
        if w not in stop_words:
            filtered_sentence.append(w)

    print(filtered_sentence)
    texts[i] = " ".join(filtered_sentence)
    texts[i] = ''.join([j for j in texts[i] if not j.isdigit()])
    texts[i] = " ".join(w for w in nltk.wordpunct_tokenize(texts[i]) if w.lower() in words or not w.isalpha())
    i+=1

#vectorizing text values
cv=CountVectorizer(stop_words='english')
x=cv.fit_transform(texts)

values=x.toarray()

attributes=cv.get_feature_names_out()

# organizing values
dataframe=pd.DataFrame(values,columns=attributes)

Y=data['polarity'].values 
X=dataframe

# categorizing values
for i in range(len(Y)):
    if Y[i] < 0:
        Y[i] = -1
    if Y[i] >= 0 and Y[i] <= 0.025:
        Y[i] = 0
    if  Y[i] >= 0.026 and Y[i] <= 1:
        Y[i] = 1

#---------------------------------------  test train split ----------------------------------
train_x,test_x,train_y,test_y = train_test_split(X,Y,test_size = 0.45)

rfc = GaussianNB()
rfc.fit(train_x,train_y)



# ---------------------------------- model evaluation -------------------------------------
y_pred = rfc.predict(test_x)

print("predicted values are ->> ",y_pred)

matrix = confusion_matrix(test_y,y_pred)

disp = ConfusionMatrixDisplay.from_predictions(test_y,y_pred)