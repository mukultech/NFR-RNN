import pandas as pd
import multi_rnn as multirnn
import multi_gru as mgru

import preprocess as pp
import lemmatization as lema
import number_to_word as ntw

train_data=pd.read_csv('training_data_double.csv')
train_data=train_data.drop(['ProjectID'], axis=1)
print('Number of words before data preprocessing: ',train_data['RequirementText_string'].apply(lambda x: len(x.split(' '))).sum())

train_data=pp.preprocess(train_data)
print('Number of words after data preprocessing: ', train_data['RequirementText_string'].apply(lambda x: len(x.split(' '))).sum())

#train_data.to_csv('train_prep.csv',index = False)
train_data=lema.lemtext(train_data)
print('Number of words after lemmatization: ',train_data['RequirementText_string'].apply(lambda x: len(x.split(' '))).sum())

train_data=ntw.numbertoword(train_data)
print('Number of words after number conversion: ',train_data['RequirementText_string'].apply(lambda x: len(x.split(' '))).sum())

#multirnn.multi_classifier(train_data)
mgru.multi_classifier(train_data)

