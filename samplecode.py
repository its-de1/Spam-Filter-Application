import wordcloud
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger') 
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
from subprocess import check_output
# Any results you write to the current directory are saved as output.
#SVM model for SMS classification using SMS Spam COllection Dataset (Kaggle)
import pandas as pd
import re
from nltk.corpus import stopwords
from sklearn import svm
email=pd.read_csv("datasets_483_982_spam.csv",encoding='latin-1')
email=email.rename(columns = {'v1':'label','v2':'message'})
cols=['label','message']
email=email[cols]
email=email.dropna(axis=0, how='any')
#Email preprocessing
num_emails=email["message"].size
def email_processing(raw_email):
    letters_only=re.sub("[^a-zA-Z]"," ",raw_email)
    words=letters_only.lower().split()
    stops=set(stopwords.words("english"))
    m_w=[w for w in words if not w in stops]
    return (" ".join(m_w))

clean_email=[]
for i in range(0,num_emails):
    clean_email.append(email_processing(email["message"][i]))

#Create new dataframe column
email["Processed_Msg"]=clean_email
cols2=["Processed_Msg","label"]
email=email[cols2]

#Create train and test sets
X_train=email["Processed_Msg"][:4000]
Y_train=email["label"][:4000]
X_test=email["Processed_Msg"][4001:4200]
Y_test=email["label"][4001:4200]

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
vectorizer=CountVectorizer(analyzer = "word",tokenizer = None,preprocessor = None,stop_words = None,max_features = 5000)

train_data_features=vectorizer.fit_transform(X_train)
train_data_features=train_data_features.toarray()

test_data_features=vectorizer.transform(X_test)
test_data_features=test_data_features.toarray()

#SVM with linear kernel
clf=svm.SVC(kernel='linear',C=1)
#clf=svm.SVC(kernel='rbf',gamma='auto',C=1000)
print ("Training")
clf.fit(train_data_features,Y_train)
print ("Testing")
predicted=clf.predict(test_data_features)
accuracy=np.mean(predicted==Y_test)
print ("Accuracy: ",accuracy)


#Validate File and Uploading Back to cloud

import csv
upload_csv = open("ClassifiedCSV.csv","wt")
uploadcsv=csv.writer(upload_csv)

print(upload_csv.closed)

with open('validate_final.csv') as csv_file:
    csv_reader = csv.reader(csv_file)
    for row in csv_reader:
      validation_data=vectorizer.transform(row)
      validation_data=validation_data.toarray()
      print ("SMS: ",row[0])
      classification=clf.predict(validation_data)
      print ("Classification: ",classification[0])
      uploadcsv.writerow([row[0],classification[0]])
upload_csv.close()
