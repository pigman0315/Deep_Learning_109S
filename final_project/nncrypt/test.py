import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def binary(x, bits):
    mask = 2**torch.arange(bits).to(x.device, x.dtype)
    return x.unsqueeze(-1).bitwise_and(mask).ne(0).byte()
def test(args,hp,alice,bob,eve):
	alice.eval();bob.eval();eve.eval()

	l1_loss_fn = nn.L1Loss()
	x = torch.arange(2**min(16,hp.data.N))
	data = []
	idx = 0
	for d in x:
		if(d == 2**idx or d == 0):
			data.append(d)
			if(d != 0):
				idx += 1
	data = torch.tensor(data)
	plain_list = binary(data,hp.data.N).cuda()
	key = torch.randint(0, 2, (1,1,hp.data.N)).cuda()

	cipher_list = []
	for plain in plain_list:
		plain = plain.view(1,1,hp.data.N).float()
		key = key.float()
		cipher = alice(plain,key)
		plain = plain.view(hp.data.N)
		cipher = cipher.view(hp.data.N)
		cipher_list.append(cipher.cpu().detach().numpy())
		print(plain.cpu().numpy())
		print(cipher.cpu().detach().numpy())
		print('---')

	cnt = 0
	c0 = cipher_list[0].copy()
	cipher_list.pop(0)
	for c in cipher_list:
		for i in range(len(c)):
			if(c[i]>c0[i]):
				cnt +=1
	print('Avg diff:', hp.data.N-cnt/len(cipher_list))