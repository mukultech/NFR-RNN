from nltk import word_tokenize
from num2words import num2words

def numbertoword(train_data):
    def numbtoword(sent):
             token_words=word_tokenize(sent)
             stem_sentence=[]
             for word in token_words:
                 if word.isdigit():
                  stem_sentence.append(num2words(word))
                  stem_sentence.append(" ")
                 else:
                  stem_sentence.append(word)
                  stem_sentence.append(" ")
                  
             sentence=""
             sentence=sentence.join(stem_sentence)
             return sentence
     #for sent in train_data.RequirementText_string:
      # st=stemsent(sent)
       
    train_data['RequirementText_string'] = train_data['RequirementText_string'].apply(numbtoword)
    return train_data
