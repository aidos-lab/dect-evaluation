import torch
import numpy as np
import pywt
import ptwt  # use "from src import ptwt" for a cloned the repo

# generate an input of even length.
data = np.array([1,1,-1,-1])
data_torch = torch.from_numpy(data.astype(np.float32))
wavelet = pywt.Wavelet('haar')

# compare the forward fwt coefficients
# print(pywt.wavedec(data, wavelet, mode='zero', level=2))
print(ptwt.wavedec(data_torch, wavelet, mode='zero', level=1))

# # invert the fwt.
# print(ptwt.waverec(ptwt.wavedec(data_torch, wavelet, mode='zero'),
#                    wavelet))