from configurations.losses.loss_functions import cross_entropy

def training_loss(dataset, logits, labels):
  return evaluation_loss(dataset, logits, labels)
  
def evaluation_loss(dataset, logits, labels):
  labels1, true_labels = dataset.split_labels(labels)
  loss = cross_entropy(logits, true_labels)
  return loss