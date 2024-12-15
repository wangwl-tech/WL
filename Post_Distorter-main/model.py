import torch
import torch.nn as nn
import torch.nn.functional as F

class Post_distorter(nn.Module):
  def __init__(self, input_dim=2, output_dim=1, hidden_dim=64):
    super(Post_distorter, self).__init__()
    self.bi_nonlinear = nn.Sequential(
      nn.Linear(input_dim, hidden_dim),
      nn.Tanhshrink()
    )
    self.mo_nonlinear_bef = nn.Sequential(
      nn.Linear(1, hidden_dim//2),
      nn.Tanhshrink()
    )
    self.mo_nonlinear_cur = nn.Sequential(
      nn.Linear(1, hidden_dim//2),
      nn.Tanhshrink()
    )
    self.ffn = nn.Sequential(
      nn.Linear(hidden_dim*2, hidden_dim, bias=False),
      nn.Sigmoid(),
      nn.Linear(hidden_dim, 1, bias=False),
      nn.Sigmoid()
    )    
  def forward(self, bef_val, cur_val):
    data = torch.cat([bef_val, cur_val], dim=1)
    combined = self.bi_nonlinear(data)
    bef = self.mo_nonlinear_bef(bef_val)
    cur = self.mo_nonlinear_cur(cur_val)
    in_ffn = torch.cat([combined, bef, cur], dim=1)
    pred = self.ffn(in_ffn)
    return pred

if __name__ == "__main__":
  post_dist = Post_distorter()
  data1 = torch.randn(3,1)
  data2 = torch.randn(3,1)
  output = post_dist(data1, data2)
  print(output)
  # print("output shape: ", output.shape)
