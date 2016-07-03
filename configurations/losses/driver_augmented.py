import loss_functions

def training_loss(dataset, logits, labels):
  logits1, true_logits = logits
  labels1, true_labels = dataset.split_labels(labels)

  training_loss = evaluation_loss(true_logits, true_labels)
  return training_loss
  
def evaluation_loss(dataset, logits, labels):
  return loss_functions.cross_entropy(logits, labels)