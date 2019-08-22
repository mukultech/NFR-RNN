from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer


def lemtext(train_data):
     lemmatizer=WordNetLemmatizer()      
     def lemsent(sent):
             token_words=word_tokenize(sent)
             stem_sentence=[]
             for word in token_words:
               stem_sentence.append(lemmatizer.lemmatize(word))
               stem_sentence.append(" ")

             sentence=""
             sentence=sentence.join(stem_sentence)
             sentence=sentence.rstrip()
             
             return sentence
             
     #for sent in train_data.RequirementText_string:
      # st=stemsent(sent)
       
     train_data['RequirementText_string'] = train_data['RequirementText_string'].apply(lemsent)
    
     #print (st)
     return train_data

