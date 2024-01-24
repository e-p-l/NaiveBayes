import numpy as np

class NaiveBayes:
  def __init__(self):
    self.priors = None
    self.likelihoods = None
    self.x_train = None
    self.y_train = None

  def compute_priors(self, labels):
    #6 possible labels
    priors = np.array([0,0,0,0,0,0])

    for label in labels:
        priors[label] += 1

    priors = priors / len(labels)

    return priors / len(labels)

  def compute_likelihoods(self, features, labels):
    # add a row of 1's for every label, to make sure the posterior is never 0
    for label in np.unique(labels):
      features = np.vstack((features, np.ones(features.shape[1])))
      labels.append(label)

    num_of_entries = features.shape[0]
    num_of_features = features.shape[1]

    unique_labels, occurences_by_label= np.unique(labels, return_counts=True)
    likelihoods = np.zeros((unique_labels.size, num_of_features))

    for i in range(num_of_features):
      for j in range(num_of_entries):
        feature = features[j,i]
        if(feature == 1):
          likelihoods[labels[j],i] += 1 / occurences_by_label[labels[j]]

    return likelihoods

  def fit(self, features, labels):
    self.x_train = features
    self.y_train = labels

    self.priors = self.compute_priors(labels)
    self.likelihoods = self.compute_likelihoods(features, labels)

  def compute_posterior(self, priors, likelihoods, entry, unique_labels):
    posterior = []

    for label_index in range(len(unique_labels)):
      posterior.append(priors[label_index])
      for feature_index in range(len(entry)):
        #likelihood of entry = 1
        if entry[feature_index] == 1:
          posterior[label_index] *= likelihoods[label_index, feature_index]
        #likelihood of entry = 0
        else:
          posterior[label_index] *= 1 - likelihoods[label_index, feature_index]

    #normalize the posterior
    normalized_posterior = posterior / np.sum(posterior)

    return normalized_posterior

  def predict(self, entries):
    predictions = []

    for entry in entries:
      posterior = self.compute_posterior(self.priors, self.likelihoods, entry, np.unique(self.y_train))
      predictions.append(posterior)

    return predictions

  def compute_accuracy(self, predictions, truths):
    num_of_accurate_predictions = 0

    for index, prediction in enumerate(predictions):
      predicted_label = np.argmax(prediction)
      if predicted_label == truths[index]:
        num_of_accurate_predictions += 1

    return num_of_accurate_predictions / len(truths)