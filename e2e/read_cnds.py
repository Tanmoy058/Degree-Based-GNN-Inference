import pickle
import torch


file_name = '../cnds_FR_P.pkl'
with open(file_name, 'rb') as f:
     cnds = pickle.load(f)
     
print('FR P SAGE')
cnd = torch.squeeze(cnds[0,:,:])
for i in range(cnd.shape[0]):
    print(cnd[i,:])
print('\n')
print('FR P GAT')
cnd = torch.squeeze(cnds[1,:,:])
for i in range(cnd.shape[0]):
    print(cnd[i,:])
print('\n')
print('\n')
file_name = '../cnds_FR_B.pkl'
with open(file_name, 'rb') as f:
     cnds = pickle.load(f)
     
print('FR B SAGE')
cnd = torch.squeeze(cnds[0,:,:])
for i in range(cnd.shape[0]):
    print(cnd[i,:])
print('\n')
print('FR B GAT')
cnd = torch.squeeze(cnds[1,:,:])
for i in range(cnd.shape[0]):
    print(cnd[i,:])
print('\n')
print('\n')
print('\n')
print('\n')



file_name = '../cnds_PA_P.pkl'
with open(file_name, 'rb') as f:
     cnds = pickle.load(f)
     
print('PA P SAGE')
cnd = torch.squeeze(cnds[0,:,:])
for i in range(cnd.shape[0]):
    print(cnd[i,:])
print('\n')
print('PA P GAT')
cnd = torch.squeeze(cnds[1,:,:])
for i in range(cnd.shape[0]):
    print(cnd[i,:])
print('\n')
print('\n')
file_name = '../cnds_PA_B.pkl'
with open(file_name, 'rb') as f:
     cnds = pickle.load(f)
     
print('PA B SAGE')
cnd = torch.squeeze(cnds[0,:,:])
for i in range(cnd.shape[0]):
    print(cnd[i,:])
print('\n')
print('PA B GAT')
cnd = torch.squeeze(cnds[1,:,:])
for i in range(cnd.shape[0]):
    print(cnd[i,:])
