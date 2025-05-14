import pickle
import torch



file_name = 'cnds_FR_P_SAGE.pkl'
with open(file_name, 'rb') as f:
    cnds = pickle.load(f)
cnd_dict = {}
cnd = torch.squeeze(cnds[0,:,:])

inds= torch.sort(cnd[:,3], descending=True)[1]
cnd = cnd[inds]
#print(cnd)

for i in range(cnd.shape[0]):
    cnd_dict[cnd[i,0].item()] = cnd[i]

#print(cnd_dict)
cnd_list = []
for cnd in cnd_dict.values():
    cnd_list.append(cnd)
    
cnds0 = torch.stack(cnd_list, dim=0)
print(cnds0)



file_name = 'cnds_FR_P_GAT.pkl'
with open(file_name, 'rb') as f:
    cnds = pickle.load(f)
cnd_dict = {}
cnd = torch.squeeze(cnds[0,:,:])

inds= torch.sort(cnd[:,3], descending=True)[1]
cnd = cnd[inds]
#print(cnd)

for i in range(cnd.shape[0]):
    cnd_dict[cnd[i,0].item()] = cnd[i]

#print(cnd_dict)
cnd_list = []
for cnd in cnd_dict.values():
    cnd_list.append(cnd)
    
cnds1 = torch.stack(cnd_list, dim=0)
print(cnds1)

cnds = torch.stack([cnds0, cnds1], dim=0)
print(cnds.shape)
print(cnds)
with open ('cnds_FR_P.pkl', 'wb') as f:
    pickle.dump(cnds, f)
    
    
    
    
    
    
    
    
    
    
    
file_name = 'cnds_PA_P_SAGE.pkl'
with open(file_name, 'rb') as f:
    cnds = pickle.load(f)
cnd_dict = {}
cnd = torch.squeeze(cnds[0,:,:])

inds= torch.sort(cnd[:,3], descending=True)[1]
cnd = cnd[inds]
#print(cnd)

for i in range(cnd.shape[0]):
    cnd_dict[cnd[i,0].item()] = cnd[i]

#print(cnd_dict)
cnd_list = []
for cnd in cnd_dict.values():
    cnd_list.append(cnd)
    
cnds0 = torch.stack(cnd_list, dim=0)
print(cnds0)



file_name = 'cnds_PA_P_GAT.pkl'
with open(file_name, 'rb') as f:
    cnds = pickle.load(f)
cnd_dict = {}
cnd = torch.squeeze(cnds[1,:,:])

inds= torch.sort(cnd[:,3], descending=True)[1]
cnd = cnd[inds]
#print(cnd)

for i in range(cnd.shape[0]):
    cnd_dict[cnd[i,0].item()] = cnd[i]

#print(cnd_dict)
cnd_list = []
for cnd in cnd_dict.values():
    cnd_list.append(cnd)
    
cnds1 = torch.stack(cnd_list, dim=0)
print(cnds1)

cnds = torch.stack([cnds0, cnds1], dim=0)
print(cnds.shape)
print(cnds)
with open ('cnds_PA_P.pkl', 'wb') as f:
    pickle.dump(cnds, f)  
