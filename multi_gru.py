def multi_classifier(train_data):
       import numpy as np
       import pandas as pd
       import matplotlib.pyplot as plt
       import plotting_data as pltdt
       import multi_pred_and_evaluation as pred_eva
       from keras.utils import to_categorical

	## set the random number seed for reproducability in result ##
       seed = 7
       np.random.seed(seed)
       from tensorflow import set_random_seed
       set_random_seed(2)

	## Drop the functional requirements ##
       classes=train_data.Class
       num=[]
       for i,Class in enumerate(classes):
          if Class=='F':
            num.append(i)
       train_data=train_data.drop(train_data.index[num])
       train_data.to_csv('train_multi.csv',index = False)
       text_1=train_data.RequirementText_string
       print ('Types of Unique Classes',train_data.Class.unique())

	## load the training dataset ##
       train_data=pd.read_csv('train_multi.csv')
       #print ('Before dropping test data: ',train_data.shape)
       length_data=len(train_data.index)
       #print (length_data)
       #test_data=train_data.sample(frac=0.1,random_state=200)
       test_data=pd.read_csv('testdata.csv')
       #train_data=train_data.drop(test_data.index)
       #print('Training data shape:',train_data.shape)
       
       #print('Testing data shape:',test_data.shape)
       dic={'A':0,'FT':1,'L':2,'LF':3,'MN':4,'O':5,'PE':6,'SC':7,'SE':8,'US':9}
       #print (dic)
       
       plt.figure(figsize=(4,4))
       train_data.Class.value_counts().plot(kind='bar');
       #plt.show()
       #print('Training data shape:',train_data.shape)
       #print(train_data.isnull().sum())

       def sent_tokin(train_data,test_data,NUM_WORDS):       
           from keras.preprocessing.text import Tokenizer
           texts=train_data.RequirementText_string
           tokenizer = Tokenizer(num_words=NUM_WORDS,lower=True)
           tokenizer.fit_on_texts(texts)
           sequences_train = tokenizer.texts_to_sequences(texts)
           sequences_test=tokenizer.texts_to_sequences(test_data.RequirementText_string)
           
           word_index = tokenizer.word_index
           return texts,word_index,sequences_train,sequences_test

       NUM_WORDS=25000
       texts,word_index,sequences_train,sequences_test=sent_tokin(train_data,test_data,NUM_WORDS)
       #print('Found %s unique tokens.' % len(word_index))

	## Pad the training dataset ###
       length=len(sequences_train)
       lenth=[]
       for i in range(length):
           #print('seq train shape', len(sequences_train[i]))
           lenth.append(len(sequences_train[i])) 
       max_length=np.max(lenth)
       #print ('maxlength :',max_length)
       #print ('First sentence after tokenizing: ',sequences_train[1])
       #print ('Training data shape:',len(sequences_train))
       #print ('First sentence: ',texts[1])
       
       for i in range(length):
         if (len(sequences_train[i]) < max_length):
             #print ('sequence length=',len(sequences_train[i]))
             for j in range(47-len(sequences_train[i])):
                sequences_train[i].append(0)
       #print ('First sentence after tokenizing and zero padding: ',sequences_train[1])
       sequences_train=np.array(sequences_train)
       
       
       #print ('First sentence after tokenizing and zero padding: ',sequences_train[1])
       #print ('Length of First sentence after tokenizing and zero padding:',len(sequences_train[1]))
       #print ('train_data.index', train_data.index)   
       
       num_labels=len(train_data.Class)
       #print (num_labels)
       for i in range(length_data):
           if i in train_data.index:
             d=train_data.Class[i]
             train_data.Class[i]=dic[d]
           
       
       #y_train=train_data.Class.apply(lambda x:dic[x])
      
       y_train = train_data.Class.astype(np.int64) #apply(lambda x:dic[x])
       #print(type(y_train))
       #print ('y_train',y_train)
       #print('Shape of training sequence :', sequences_train.shape)
       #print('Shape of training labels', y_train.shape)
       
	## Pad the testing dataset ##
       length=len(sequences_test)
       print (length)       
       
       for i in range(length):
         if (len(sequences_test[i]) < max_length):
             for j in range(47-len(sequences_test[i])):
                sequences_test[i].append(0)
       
       sequences_test=np.array(sequences_test)
       #print ('First sentence of testing dataset after tokenizing and zero padding: ',sequences_test[1])
       
       #print ('Testing data shape:',len(sequences_test))
       
       '''
       for key, value in dic.items():       
           print (key ,'=' ,dic[key])
           
       '''
       num_labels=len(test_data.Class)
       print (num_labels)

       for i in range(length_data):
         if i in test_data.index:  
           d=test_data.Class[i]
           test_data.Class[i]=dic[d]
         
       y_test = test_data.Class.astype(np.int64) #apply(lambda x:dic[x])
              
       #print('Shape of testing sequence :', sequences_test.shape)
       #print('Shape of testing labels', y_test.shape)
       
	## Word2vec on the dataset ##

       def word2vec_embed(text_1,word_index,NUM_WORDS,embedding_dim):

           from gensim.models.keyedvectors import KeyedVectors
           EMBEDDING_DIM = embedding_dim
           word_vectors = KeyedVectors.load_word2vec_format('/home/softlab/Desktop/python/arif/GoogleNews-vectors-negative300.bin', binary=True)
           print('Found %s word vectors of word2vec' % len(word_vectors.vocab))
           vocabulary_size=min(len(word_index)+1,NUM_WORDS)
           print ('Num_words: ',NUM_WORDS)
           print ('len_wi:' ,len(word_index))
           print ('vocabulary size',vocabulary_size)
           embedding_matrix = np.zeros((vocabulary_size, EMBEDDING_DIM))
           for word, i in word_index.items():
               if word in word_vectors.vocab:
                 embedding_matrix[i] = word_vectors.word_vec(word)
                
           #print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))
           return vocabulary_size, embedding_matrix

	## Model Building ##
       EMBEDDING_DIM= 300
       vocabulary_size,embedding_matrix=word2vec_embed(text_1,word_index,NUM_WORDS,EMBEDDING_DIM)
       from keras.layers import  Dense, Embedding,LSTM, GRU
       from keras.optimizers import Adam
       from keras.models import Sequential
       from keras.layers import LeakyReLU
       from sklearn.utils import class_weight
       from sklearn.model_selection import StratifiedKFold
       from sklearn.metrics import confusion_matrix
       from sklearn.metrics import accuracy_score
       from sklearn.metrics import classification_report
       from numpy import array

       sequence_length = max_length #X_train.shape[1] # from (1245,47) 47 is the 1
       #print (sequence_length)
       class_weight=class_weight.compute_class_weight('balanced',np.unique(y_train),y_train)
       #print (class_weight)

	## K - fold crossvalidation ##
       kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
       cvscores = []
       test_scores = []
       i=0

       for train, validation in kfold.split(sequences_train, y_train):
         #print ('Training data shape',  train.shape)
         #print ('Validation data shape' , validation.shape)
         print ('Fold number :',i)
         
         model = Sequential()
         model.add(Embedding(vocabulary_size,EMBEDDING_DIM,weights=[embedding_matrix], input_length=sequence_length,trainable=False))
         model.add(GRU(units=300,dropout=0.2,recurrent_dropout=0.2))
         model.add(Dense(units=300))
         model.add(LeakyReLU(alpha=0.1))
         model.add(Dense(units=10,activation='softmax'))
         
         print (model.summary())
         adam = Adam(lr=0.001,beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
         model.compile(loss='categorical_crossentropy',optimizer=adam,metrics=['acc'])

	## Training the model ##
         Y_train=to_categorical(y_train,num_classes=10)
         history = model.fit(array(sequences_train[train]), Y_train[train], batch_size=50, epochs=200, validation_data=(sequences_train[validation], Y_train[validation]), verbose=0,class_weight=class_weight)  # starts training
         history_dict = history.history
         #print(history_dict.keys())
         scores = model.evaluate(sequences_train[validation], Y_train[validation], verbose=0)
         #print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
         cvscores.append(scores[1] * 100)
         #print ('scores',scores)
         
         ypred=model.predict(sequences_train[validation])
         ypred2=np.argmax(ypred, axis=1)
         y_test2=[]
         lengt=len(Y_train[validation])
         for j in range(lengt):
           y_test2.append(np.argmax(Y_train[validation][j]))
         #print (dic)
         #print ('Actual values =',y_test2)
         #print ('Predicted values= ',ypred2)
         my_tags= ['A','FT','L','LF','MN','O','PE','SC','SE','US']
         cm = confusion_matrix(y_test2,ypred2)
         cm_df=pd.DataFrame(cm,index=['A','FT','L','LF','MN','O','PE','SC','SE','US'],
                            columns=['A','FT','L','LF','MN','O','PE','SC','SE','US'])
         print (cm_df)
         test_score=accuracy_score(y_test2,ypred2)
         print('Validation prediction accuracy is %s' % test_score)
         test_scores.append(test_score*100)
         print(classification_report( y_test2,ypred2,target_names=my_tags))
         
         y_pred=model.predict(sequences_test)
         y_pred_2=np.argmax(y_pred, axis=1)
         y_test_2=[]
         y_test_2=np.asarray(y_test)
         #print (dic)
         #print ('Actual values =',y_test_2)
         #print ('Predicted values= ',y_pred_2)
         my_tags= ['A','FT','L','LF','MN','O','PE','SC','SE','US']
         cm = confusion_matrix(y_test_2,y_pred_2)
         cm_df=pd.DataFrame(cm,index=['A','FT','L','LF','MN','O','PE','SC','SE','US'],
                            columns=['A','FT','L','LF','MN','O','PE','SC','SE','US'])
         print (cm_df)
         test_score=accuracy_score(y_test_2,y_pred_2)
         print('Testing prediction accuracy is: %s' % test_score)
         test_scores.append(test_score*100)
         print(classification_report( y_test_2,y_pred_2,target_names=my_tags))
         #pred_eva.pred_and_evaluation(model,X_test,y_test,X_val, y_val,dic)

	## Plot the evaluation ##
         #pltdt.plotting_data(history_dict)
         i=i+1
       print("cross validation accuracy %.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
       print("Overall testing accuracy %.2f%% (+/- %.2f%%)" % (np.mean(test_scores), np.std(test_scores)))
       
       


