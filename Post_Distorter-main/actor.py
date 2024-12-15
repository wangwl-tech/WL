class BaseActor(object):
  def __init__(self, net, objective):
    self.net = net
    self.objective = objective
  def __call__(self, data):
    raise NotImplementedError
  def train(self, mode=True):
    self.net.train(mode)

class Led_Actor(BaseActor):
  def __init__(self, net, objective):
    super().__init__(net, objective)
  def __call__(self, bef_val, cur_val, label):
    pred = self.net(bef_val, cur_val)
    return self.objective(pred, label)