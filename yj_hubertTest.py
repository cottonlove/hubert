import torch, torchaudio

'''
Load checkpoint (either hubert_soft or hubert_discrete)
hubert = torch.hub.load("bshall/hubert:main", "hubert_discrete", trust_repo=True).cuda() 
이러면 Using cache found in /home/cottonlove/.cache/torch/hub/bshall_hubert_main 에 있는 코드랑 checkpoint들고옴
그래서 내가 코드 아무리 바꿔도 반영 안된다.
그래서 저기 checkpoint 카피하고 hub대신 load로 direct하게 들고옴
'''


import sys
sys.path.append('/home/cottonlove/hubert')

from hubert import HubertDiscrete, HubertSoft, HubertSSL, Hubert
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present
from sklearn.cluster import KMeans
import numpy as np
import torch.nn.functional as F

import librosa

'''code for extracting discrete features'''
# hubert_weight = torch.load('/home/cottonlove/hubert/checkpoints/hubert-discrete-96b248c5.pt')
# kmeans = kmeans100(pretrained=True)
# hubert = HubertDiscrete(kmeans)
# consume_prefix_in_state_dict_if_present(hubert_weight["hubert"], "module.") #To let a non-DDP model load a state dict from a DDP model, consume_prefix_in_state_dict_if_present() needs to be applied to strip the prefix “module.” in the DDP state dict before loading.
# hubert.load_state_dict(hubert_weight["hubert"])
# hubert.eval()
# hubert = hubert.cuda() #cuda로 안올려주면 input이랑 모델 weight가 같은 type 아니라고 error 뜬다.

# # #print("type of hubert: ", hubert)

# # # # Load audio
# source, sr = torchaudio.load("/home/cottonlove/baseline_code/DB/LibriTTS/train-clean-100/19/198/19_198_000000_000002.wav")
# source = torchaudio.functional.resample(source, sr, 16000)
# source = source.unsqueeze(0).cuda()

# units = hubert.units(source)
# print("units: ", units)  #layer7기준: 265, layer6기준: 265

#print("source shape: ", source.shape) #source shape:  torch.Size([1, 1, 84800])

# # Extract speech units
# units = hubert.units(source)
# print("units.shape: ", units.shape)  #layer7기준: 265, layer6기준: 265

# torch.save(units, 'units_layer7.pt')
# # torch.save(units, 'units_layer6.pt')

# # Reload tensors from files
# # units_layer7 = torch.load('units_layer7.pt')
# # units_layer6 = torch.load('units_layer6.pt')

# # Compare if the tensors are equal
# # if torch.equal(units_layer7, units):
# #     print("The tensors are the same.") #The tensors are the same.
# # else:
# #     print("The tensors are different.")


'''code for extracting SSL features from pretrained HuBERT'''
#checkpoint = torch.load('/home/cottonlove/hubert/checkpoints/hubert-discrete-96b248c5.pt') 
# checkpoint2 = torch.load('/home/cottonlove/hubert/checkpoints/hubert-soft-35d9f29f.pt')
# hubert = Hubert()
# consume_prefix_in_state_dict_if_present(checkpoint2["hubert"], "module.") #To let a non-DDP model load a state dict from a DDP model, consume_prefix_in_state_dict_if_present() needs to be applied to strip the prefix “module.” in the DDP state dict before loading.
# del checkpoint2["hubert"]["label_embedding.weight"]
# del checkpoint2["hubert"]["proj.weight"]
# del checkpoint2["hubert"]["proj.bias"]
# hubert.load_state_dict(checkpoint2["hubert"], strict=False)
# hubert.eval()
# hubert = hubert.cuda() #cuda로 안올려주면 input이랑 모델 weight가 같은 type 아니라고 error 뜬다.

# checkpoint = torch.load('/home/cottonlove/hubert/checkpoints/hubert-discrete-96b248c5.pt') 
# hubert = HubertSSL()
# consume_prefix_in_state_dict_if_present(checkpoint["hubert"], "module.")
# # del checkpoint["hubert"]["label_embedding.weight"]
# # del checkpoint["hubert"]["proj.weight"]
# # del checkpoint["hubert"]["proj.bias"]
# hubert.load_state_dict(checkpoint["hubert"])
# hubert.eval()
# hubert = hubert.cuda()

# #print("type of hubert: ", hubert)

# # # # Load audio
# source, sr = torchaudio.load("/home/cottonlove/baseline_code/DB/LibriTTS/train-clean-100/19/198/19_198_000000_000002.wav")
# source = torchaudio.functional.resample(source, sr, 16000)
# source = source.unsqueeze(0).cuda()

# # # print("source shape: ", source.shape) #source shape:  torch.Size([1, 1, 84800])
# source = F.pad(source, ((400 - 320) // 2, (400 - 320) // 2))
# SSLfeatures = hubert.encode(source,layer=6)[0]
# # SSLfeatures = hubert.SSLfeatures(source)

# # print("SSLfeatures.shape: ", SSLfeatures)
# # print(SSLfeatures)
# torch.save(SSLfeatures, 'softvc_hubertsoft_SSLfeatures_6.pt')

# softvc_hubertsoft_SSLfeatures_6.pt랑 softvc_hubertdiscrete_SSLfeatures_6.pt 비교하기
# softvc_hubertsoft_SSLfeatures_6 = torch.load('softvc_hubertsoft_SSLfeatures_6.pt').to('cuda:0')
# softvc_hubertdiscrete_SSLfeatures_6 = torch.load('softvc_hubertdiscrete_SSLfeatures_6.pt').to('cuda:0')

# print(torch.eq(softvc_hubertsoft_SSLfeatures_6.squeeze(0),softvc_hubertdiscrete_SSLfeatures_6.squeeze(0))) #false
# # 제곱 오차 계산
# difference = softvc_hubertdiscrete_SSLfeatures_6.squeeze(0) - softvc_hubertsoft_SSLfeatures_6.squeeze(0)
# squared_error = difference ** 2
# # MSE 계산
# mse = torch.mean(squared_error) 

# print("MSE:", mse.item()) #MSE: 0.006193007342517376
# print(torch.allclose(softvc_hubertdiscrete_SSLfeatures_6,softvc_hubertsoft_SSLfeatures_6.squeeze(0),atol=1e-5)) #False

# softvc_SSLfeatures_6 = torch.load('softvc_SSLfeatures_6.pt').to('cuda:0')
# print(softvc_SSLfeatures_6)
# # # # #Compare if the tensors are equal
# if torch.equal(softvc_hubert_SSLfeatures_6, softvc_SSLfeatures_6):
#     print("The tensors are the same.") 
# else:
#     print("The tensors are different.") #The tensors are different.

#print(np.allclose(softvc_hubert_SSLfeatures_6.detach().cpu().numpy(), softvc_SSLfeatures_6.detach().cpu().numpy())) #False

###### cached 된거랑 내가 만든거랑 왜 차이가 나죠
# Reload tensors from files
# ssl_layer7 = torch.load('softvc_SSLfeatures_7.pt').to('cuda:0') # 밑에 torch.equal로 비교할때 다른 device에 있음 에러남
# print("ssl_layer7.shape: ",ssl_layer7.shape) #torch.Size([1, 265, 768]
# ssl_layer6 = torch.load('softvc_SSLfeatures_6.pt').to('cuda:0')
# print("ssl_layer6.shape: ",ssl_layer6.shape) #torch.Size([1, 265, 768])
# style_ssl_layer6 = torch.load('/home/cottonlove/baseline_code/stylebook/stylebook_layer6_ssl.pt').to('cuda:0')
# print(style_ssl_layer6.shape) #[265, 768]
# print(SSLfeatures.shape) #[1, 265, 768]
# print(softvc_hubertsoft_SSLfeatures_6.squeeze(0))

# print(softvc_hubert_SSLfeatures_6)
# print(SSLfeatures==softvc_hubert_SSLfeatures_6) # True. Hubert()랑 HubertSSL() 둘다 같음

# print(softvc_hubert_SSLfeatures_6.squeeze(0))
# print(style_ssl_layer6)

# print(torch.eq(softvc_hubertdiscrete_SSLfeatures_6.squeeze(0),style_ssl_layer6)) #false
# # 제곱 오차 계산
# difference = softvc_hubertdiscrete_SSLfeatures_6.squeeze(0) - style_ssl_layer6
# squared_error = difference ** 2
# # MSE 계산
# mse = torch.mean(squared_error) 

# print("MSE:", mse.item()) #MSE: 5.816733813363006e-14

# print(torch.allclose(softvc_hubertdiscrete_SSLfeatures_6,style_ssl_layer6,atol=1e-5)) #True
# print(torch.allclose(softvc_hubertdiscrete_SSLfeatures_6,style_ssl_layer6,atol=1e-6)) #False

# print(softvc_hubert_SSLfeatures_6.squeeze(0)[0])
# print(style_ssl_layer6[0])
# print(softvc_hubert_SSLfeatures_6.squeeze(0)[0]==style_ssl_layer6[0])

# print(np.allclose(SSLfeatures.detach().cpu().numpy(), style_ssl_layer6.detach().cpu().numpy()))



# style_ssl_layer6_woPAD = torch.load('/home/cottonlove/baseline_code/stylebook/stylebook_layer6_ssl_woPAD.pt').to('cuda:0')

# stylebook_layer6_ssl_100 = torch.load('/home/cottonlove/baseline_code/stylebook/stylebook_layer6_ssl_100.pt').to('cuda:0')
# print(style_ssl_layer6)
# print(stylebook_layer6_ssl_100)

# print(style_ssl_layer6==stylebook_layer6_ssl_100) #same

# kmeans_ssl_layer6 = KMeans(n_clusters=100).fit(ssl_layer6.squeeze(0).cpu())
# kmeans_style_ssl_layer6 = KMeans(n_clusters=100).fit(style_ssl_layer6.squeeze(0).cpu())

# print(kmeans_ssl_layer6.labels_.shape) # (265,)
# print(kmeans_style_ssl_layer6.labels_.shape) # (265,)
# print(kmeans_ssl_layer6.cluster_centers_)
# print(kmeans_ssl_layer6.labels_) 
# print(kmeans_style_ssl_layer6.cluster_centers_)
# print(kmeans_style_ssl_layer6.labels_) 


''' style_ssl_layer6 와 ssl_layer6간의 텐서 다름 비교'''
# difference = style_ssl_layer6 - ssl_layer6

# # 제곱 오차 계산
# squared_error = difference ** 2

# # MSE 계산
# mse = torch.mean(squared_error)

# print("MSE:", mse.item()) #MSE: 9.748920319907484e-07

''' ssl_layer7 와 ssl_layer6간의 텐서 다름 비교'''
# difference = ssl_layer7 - ssl_layer6

# # 제곱 오차 계산
# squared_error = difference ** 2

# # MSE 계산
# mse = torch.mean(squared_error)

# print("MSE:", mse.item()) MSE: 0.032734811305999756

# x1 =kmeans.predict(ssl_layer6.squeeze().cpu().numpy())
# print(x1)

# x2 =kmeans.predict(style_ssl_layer6.squeeze().cpu().numpy())
# print(x2)

# x3 = kmeans.predict(ssl_layer7.squeeze().cpu().numpy())
# print(x3)

# print(x3.shape)

# print(x1 == x2) #true

# print(x1.shape) #(265,)


# print("style_ssl_layer6.shape: ",style_ssl_layer6.shape)
# squeeze_style_ssl_layer6 = style_ssl_layer6.squeeze(0)
# print(squeeze_style_ssl_layer6==style_ssl_layer6)

# # # # #Compare if the tensors are equal
# if torch.equal(squeeze_style_ssl_layer6, ssl_layer6):
#     print("The tensors are the same.") 
# else:
#     print("The tensors are different.") #The tensors are different.




'''output of Soft Unit Encoder'''
# checkpoint2 = torch.load('/home/cottonlove/hubert/checkpoints/hubert-soft-35d9f29f.pt')
# hubert = HubertSoft()
# consume_prefix_in_state_dict_if_present(checkpoint2["hubert"], "module.") #To let a non-DDP model load a state dict from a DDP model, consume_prefix_in_state_dict_if_present() needs to be applied to strip the prefix “module.” in the DDP state dict before loading.
# # del checkpoint2["hubert"]["label_embedding.weight"]
# # del checkpoint2["hubert"]["proj.weight"]
# # del checkpoint2["hubert"]["proj.bias"]
# hubert.load_state_dict(checkpoint2["hubert"], strict=False)
# hubert.eval()
# hubert = hubert.cuda() #cuda로 안올려주면 input이랑 모델 weight가 같은 type 아니라고 error 뜬다.

# # import torch, torchaudio

# # # Load checkpoint (either hubert_soft or hubert_discrete)
# # hubert = torch.hub.load("bshall/hubert:main", "hubert_soft", trust_repo=True).cuda()

# # Load audio
# source, sr = torchaudio.load("/home/cottonlove/baseline_code/DB/LibriTTS/train-clean-100/19/198/19_198_000000_000002.wav")
# source = torchaudio.functional.resample(source, sr, 16000)
# source = source.unsqueeze(0).cuda()

# # Extract speech units
# units = hubert.units(source)

# print(units.shape) #torch.Size([1, 265, 256])


#TODO

# load HuBERT Discrete from SoftVC

checkpoint = torch.load('/home/cottonlove/hubert/checkpoints/hubert-discrete-96b248c5.pt')
# checkpoint2 = torch.load('/home/cottonlove/hubert/checkpoints/hubert-soft-35d9f29f.pt')
hubert = Hubert()
consume_prefix_in_state_dict_if_present(checkpoint["hubert"], "module.") #To let a non-DDP model load a state dict from a DDP model, consume_prefix_in_state_dict_if_present() needs to be applied to strip the prefix “module.” in the DDP state dict before loading.
del checkpoint["hubert"]["label_embedding.weight"]
del checkpoint["hubert"]["proj.weight"]
del checkpoint["hubert"]["proj.bias"]
hubert.load_state_dict(checkpoint["hubert"], strict=False)
hubert.eval()
hubert = hubert.cuda() #cuda로 안올려주면 input이랑 모델 weight가 같은 type 아니라고 error 뜬다.


'''F.pad 없이 SSL features (softVC의 hubert, stylebook의 hubert 6th layer) 비교 => 같다!'''
# # # # # Load audio
# source, sr = torchaudio.load("/home/cottonlove/baseline_code/DB/LibriTTS/train-clean-100/19/198/19_198_000000_000002.wav")
# source = torchaudio.functional.resample(source, sr, 16000)
# source = source.unsqueeze(0).cuda()

# SSLfeatures = hubert.encode(source,layer=6)[0]
# print("SSLfeatures.shape: ", SSLfeatures.shape) #torch.Size([1, 264, 768])
# torch.save(SSLfeatures, 'softvc_hubertdiscrete_SSLfeatures_6.pt') 

# style_ssl_layer6 = torch.load('/home/cottonlove/baseline_code/stylebook/stylebook_layer6_ssl_100_noPad.pt').to('cuda:0')
# softVC_ssl_layer6 = torch.load('/home/cottonlove/hubert/softvc_hubertdiscrete_SSLfeatures_6.pt').to('cuda:0')
# print("style_ssl_layer6.shape: ", style_ssl_layer6.shape) # torch.Size([264, 768])
# print("softVC_ssl_layer6.shape: ", softVC_ssl_layer6.shape) #torch.Size([1, 264, 768])

# print(torch.allclose(style_ssl_layer6,softVC_ssl_layer6, atol=1e-5)) #True


'''torchaudio.load랑 librosa.load 비교'''
# source, sr = torchaudio.load("/home/cottonlove/baseline_code/DB/LibriTTS/train-clean-100/19/198/19_198_000000_000002.wav")
# temp = source
# print(sr) #24000
# print("temp.shape: ", temp.shape) #torch.Size([1, 127200])
# print(temp)

# temp2, t = librosa.load("/home/cottonlove/baseline_code/DB/LibriTTS/train-clean-100/19/198/19_198_000000_000002.wav",sr=None)
# print("temp2.shape: ", temp2.shape) # (116865,)
# print(t) #22050(sr=None안해주면 defualt) #24000
# print(temp2)
# print(temp == temp2) # False
# print(np.allclose(temp, temp2, atol=1e-5)) #True

'''이거 resamplng 후에는 다르다고 나옴'''

lib_sr, t = librosa.load("/home/cottonlove/baseline_code/DB/LibriTTS/train-clean-100/19/198/19_198_000000_000002.wav",sr=None)
lib_sr = torch.from_numpy(lib_sr)
lib_sr = torchaudio.functional.resample(lib_sr, t, 16000)

tor_sr, sr = torchaudio.load("/home/cottonlove/baseline_code/DB/LibriTTS/train-clean-100/19/198/19_198_000000_000002.wav")
tor_sr = torchaudio.functional.resample(tor_sr, sr, 16000)
print(lib_sr)
print(lib_sr.shape)
print(tor_sr)
print(tor_sr.shape)
print(lib_sr == tor_sr)

# print("sig.shape: ", sig.shape) #sig.shape:  torch.Size([84800])
# sig = sig.unsqueeze(0)
# sig = sig.unsqueeze(0)
# print("sig.shape: ", sig.shape) #sig.shape:  torch.Size([1, 1, 84800])
# sig = sig.to("cuda:0")
# print(sig == source) #False
# print(sig)
# print(source)
# print(torch.allclose(sig, source, atol=1e-5)) #False... why?


'''5초 뽑은 걸로 앞뒤 0.05초 fade in/out 처리 후 SSL features (softVC의 hubert, stylebook의 hubert 6th layer) 비교'''

# VOCAB_SIZE = 100 #200
# SR = 16000
# SEG_LEN = 80000
# FADE_LEN = 800

# sig, _ = librosa.load("/home/cottonlove/baseline_code/DB/LibriTTS/train-clean-100/19/198/19_198_000000_000002.wav", sr=SR)
# sig = torch.Tensor(sig)


# n_seg = len(sig) // SEG_LEN      # Discard the last frame
# print("n_seg: ", n_seg) # 1

# def apply_fading(sig, fade_len):
#     fade_in = torch.arange(fade_len, device=sig.device) / fade_len
#     fade_out = torch.flipud(fade_in)

#     sig[:fade_len] *= fade_in
#     sig[-fade_len:] *= fade_out

#     return sig

# @torch.no_grad()
# def extract_mel(sig):
#     from speechbrain.lobes.models import HifiGAN

#     mel = HifiGAN.mel_spectogram(
#         sample_rate=16000, hop_length=256, win_length=1024, n_fft=1024,
#         n_mels=80, f_min=0.0, f_max=8000.0, power=1, normalized=False,
#         norm="slaney", mel_scale="slaney", compression=True,
#         audio=sig
#     ).transpose(-1, -2)

#     return mel

# for n in range(n_seg):
#     # Audio
#     seg = sig[n * SEG_LEN:(n + 1) * SEG_LEN] #5초뽑음
#     print("seg.shape: ", seg.shape)
#     seg = apply_fading(seg, FADE_LEN) #fading해줌

#     print("after fading seg.shape: ", seg.shape)
#     # Mel
#     # mel = extract_mel(seg).cpu()
#     # print("mel.shape: ", mel.shape) #([313, 80])

#     # # HuBERT VQ
#     # hubert_vq = extract_hubert_vq(seg, hubert_model)
#     # print("hubert_vq.shape: ", hubert_vq.shape) #([249, 768])

#     ## SSL features
#     seg = seg.unsqueeze(0).unsqueeze(0).to("cuda:0")
#     print("seg.shape: ", seg.shape) #seg.shape:  torch.Size([1, 1, 80000])
#     SSLfeatures = hubert.encode(seg,layer=6)[0]

#     print("SSLfeatures.shape: ", SSLfeatures.shape) #w/fading: torch.Size([1, 249, 768])
#     torch.save(SSLfeatures, 'softvc_hubertdiscrete_SSLfeatures_6_fading.pt') 

#     break

# style_ssl_layer6_faindg = torch.load('/home/cottonlove/baseline_code/stylebook/stylebook_layer6_ssl_100_fading.pt').to('cuda:0')
# softVC_ssl_layer6_fading = torch.load('/home/cottonlove/hubert/softvc_hubertdiscrete_SSLfeatures_6_fading.pt').to('cuda:0')
# print("style_ssl_layer6_faindg.shape: ", style_ssl_layer6_faindg.shape) # torch.Size([249, 768])
# print("softVC_ssl_layer6_fading.shape: ", softVC_ssl_layer6_fading.shape) # ([1, 249, 768])

# print(torch.allclose(style_ssl_layer6_faindg,softVC_ssl_layer6_fading, atol=1e-5)) # True