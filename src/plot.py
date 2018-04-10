from matplotlib import pyplot as plt


def plot_training_score(history):

    plt.figure()
    plt.title("loss")
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.plot([num for num in range(1, len(history.history['loss'])+1)],
             history.history['loss'])
    plt.axis([1, 10, min(history.history['loss']),
              max(history.history['loss'])])

    plt.figure()
    plt.title("Accuracy")
    plt.xlabel("epochs")
    plt.ylabel("acc")
    plt.plot([num for num in range(1, len(history.history['acc'])+1)],
             history.history['acc'])
    plt.axis([1, 10, min(history.history['acc']), max(history.history['acc'])])
    #print('Availible variables to plot: {}'.format(history.history.keys()))
    plt.figure()
    plt.title("loss")
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.plot([num for num in range(1, len(history.history['loss'])+1)],
             history.history['loss'])
    plt.axis([1, 10, min(history.history['loss']),
              max(history.history['loss'])])

    plt.figure()
    plt.title("Accuracy")
    plt.xlabel("epochs")
    plt.ylabel("acc")
    plt.plot([num for num in range(1, len(history.history['acc'])+1)],
             history.history['acc'])
    plt.axis([1, 10, min(history.history['acc']), max(history.history['acc'])])

    plt.show()
  # TODO: Visulize the plot, to be applied after traing is complete
