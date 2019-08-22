def plotting_data(history_dict):
              import matplotlib.pyplot as plt
              acc = history_dict['acc']
              val_acc = history_dict['val_acc']
              loss = history_dict['loss']
              val_loss = history_dict['val_loss']

              epochs = range(1, len(acc) + 1)

              # "bo" is for "blue dot"
              plt.plot(epochs, loss, 'bo', label='Training loss')
              # b is for "solid blue line"
              plt.plot(epochs, val_loss, 'b', label='Validation loss')
              plt.title('Training and validation loss')
              plt.xlabel('Epochs')
              plt.ylabel('Loss')
              plt.legend()
              plt.show()

              plt.clf()   # clear figure

              plt.plot(epochs, acc, 'bo', label='Training acc')
              plt.plot(epochs, val_acc, 'b', label='Validation acc')
              plt.title('Training and validation accuracy')
              plt.xlabel('Epochs')
              plt.ylabel('Accuracy')
              plt.legend()
              plt.show()

