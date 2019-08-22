import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
def pred_and_evaluation(model,X_test,y_test,X_val, y_val,dic):
              y_pred=model.predict(X_test)
              y_pred_2=np.argmax(y_pred, axis=1)
              y_test_2=[]
              y_test_2=np.asarray(y_test)
              print (dic)
              print ('Actual values =',y_test_2)
              print ('Predicted values= ',y_pred_2)
              my_tags= ['A','FT','L','LF','MN','O','PE','SC','SE','US']
              cm = confusion_matrix(y_test_2,y_pred_2)
              print (cm)
              print('Testing accuracy is %s' % accuracy_score(y_test_2,y_pred_2))
              print(classification_report( y_test_2,y_pred_2,target_names=my_tags))
              score = model.evaluate(X_val, y_val, verbose=1)
              print('Validation loss:', score[0])
              print('Validation accuracy:', score[1])
