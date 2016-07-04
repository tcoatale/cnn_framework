from configurations.losses.loss_functions import cross_entropy

def training_loss(dataset, logits, labels):
  return evaluation_loss(dataset, logits, labels)
  
def evaluation_loss(dataset, logits, labels):
  logits1, true_logits = logits
  labels1, true_labels = dataset.split_labels(labels)
  loss = cross_entropy(true_logits, true_labels)
  return loss