import tensorflow as tf

class SessionManager:
  def __init__(self, config):
    self.config = config

  def restore(self, saver):
      sess = tf.Session()
      ckpt = tf.train.get_checkpoint_state(self.config.ckpt_dir)
      if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
      else:
        raise ValueError('No checkpoint in this directory: ' + self.config.ckpt_dir)
      return global_step, sess
      
