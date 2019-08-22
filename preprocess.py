import re
import nltk
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords

def preprocess(train_data):
     REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]@,;]')
     BAD_SYMBOLS_RE = re.compile('[^0-9a-z ]')
     STOPWORDS = set(stopwords.words('english'))

     def clean_text(text):
         text = text.lower() # lowercase text
         text = REPLACE_BY_SPACE_RE.sub(' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text
         text = BAD_SYMBOLS_RE.sub('', text) # delete symbols which are in BAD_SYMBOLS_RE from text
         text = ' '.join(word for word in text.split() if word not in STOPWORDS) # delete stopwors from text
         text = text.strip()
         #print (text)
         return text

     train_data['RequirementText_string'] = train_data['RequirementText_string'].apply(clean_text)
     return train_data
