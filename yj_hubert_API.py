import torch, torchaudio

source, sr = torchaudio.load("/home/cottonlove/baseline_code/DB/LibriTTS/train-clean-100/19/198/19_198_000000_000002.wav")
source = torchaudio.functional.resample(source, sr, 16000)
source = source.unsqueeze(0).cuda()
print("source.shape: ", source.shape)
hubert = torch.hub.load("bshall/hubert:main", "hubert_discrete", trust_repo=True).cuda()

units = hubert.units(source)
print(units.shape)
print(units)

hubert_soft = torch.hub.load("bshall/hubert:main", "hubert_soft", trust_repo=True).cuda()
units = hubert_soft.units(source)
print(units.shape)
print(units)
# torch.save(units, 'cached_units_layer7.pt')

##
# units_layer7 = torch.load('units_layer7.pt')

# if torch.equal(units_layer7, units):
#     print("The tensors are the same.") #The tensors are same.
# else:
#     print("The tensors are different.")