import numpy as np
import scipy
import scipy.signal as sig
import matplotlib.pyplot as plt 
# Led params
def post_ditorter(P_opt, post_params):
  # np.array([c0, c1, c2, d0, d1, d2])
  estimated = np.zeros_like(P_opt)
  n_t = (2*post_params[3]*P_opt + post_params[4])**0.5 + post_params[5]
  for idx in range(P_opt.shape[0] - 1):
    if idx == 0:
      estimated[0] = 0
      continue
    estimated[idx] = post_params[0]*n_t[idx] + post_params[1]*n_t[idx - 1] + post_params[2]*(n_t[idx - 1]**2)

  # print(estimated.shape)
  return estimated

def led_model(led_params, sample_num):
  A0 = led_params[0]
  A1 = led_params[1]
  A2 = led_params[2]
  A3 = led_params[3] 
  A4 = led_params[4]
  A5 = led_params[5]

  c0 = 1/led_params[0]
  c1 = -led_params[1]/led_params[0]
  c2 = -led_params[2]/led_params[0]
  d0 = 1/led_params[5]
  d1 = (led_params[4]**2)/(4*(led_params[5]**2))
  d2 = -led_params[4]/(2*led_params[5])
  post_dist_param = np.array([c0, c1, c2, d0, d1, d2]).astype('float64') 

  N = sample_num     # sampling number 
  fs = 4e8      # sampling rate 0.4GHz
  fc = 1e6      # input frequency 1MHz
  t = np.arange(0, (N+1)/fs, 1/fs)
  s1 = (sig.square(2*scipy.pi*fc*t,0.5)+1)*0.5 # square wave
  # s1 = mapminmax(s1,0.002039,0.01571)

  n = np.zeros(N+1)
  for k in range(N):
    n[k+1] = A0*s1[k] + A1*n[k]+A2*n[k]*n[k]

  P = np.zeros(N+1)
  for k in range(N+1):
    P[k] = (A4*n[k] + A5*n[k]*n[k])*0.5

  estimated = post_ditorter(P, post_dist_param)
  print((np.abs(estimated - s1)**2).mean())

  # plot the s1 and singnal P
  t_axis = np.arange(0,s1.shape[0])
  plt.figure(1, figsize=(28,12))
  plt.plot(t_axis, P)
  plt.plot(t_axis, s1)
  plt.plot(t_axis, estimated)
  plt.savefig("./out.png")
  print("Data initialized complete, output the distort curve.")
  led_data = dict()
  led_data['input'] = s1
  led_data['output'] = P
  return led_data
if __name__ == "__main__":
  led_params = np.array([1, 0.962399, -0.003345, 0, 0.074754, 0.0055680157]).astype('float64') 
  led_model(led_params, 1000)
