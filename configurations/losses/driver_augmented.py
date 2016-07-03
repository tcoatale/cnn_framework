from configurations.losses.loss_functions import cross_entropy

def training_loss(dataset, logits, labels):
  logits1, true_logits = logits
  labels1, true_labels = dataset.split_labels(labels)

  training_loss = evaluation_loss(dataset, true_logits, true_labels)
  return training_loss
  
def evaluation_loss(dataset, logits, labels):
  return cross_entropy(logits, labels)