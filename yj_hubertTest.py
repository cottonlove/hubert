import torch, torchaudio

'''
Load checkpoint (either hubert_soft or hubert_discrete)
hubert = torch.hub.load("bshall/hubert:main", "hubert_discrete", trust_repo=True).cuda() 
이러면 Using cache found in /home/cottonlove/.cache/torch/hub/bshall_hubert_main 에 있는 코드랑 checkpoint들고옴
그래서 내가 코드 아무리 바꿔도 반영 안된다.
그래서 저기 checkpoint 카피하고 hub대신 load로 direct하게 들고옴
'''

from hubert import HubertDiscrete, HubertSoft,HubertSSL
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present
from sklearn.cluster import KMeans

def _kmeans(
    num_clusters: int, pretrained: bool = True) -> KMeans:
    kmeans = KMeans(num_clusters)
    if pretrained:
        checkpoint = torch.load("/home/cottonlove/hubert/checkpoints/kmeans100-50f36a95.pt")
        kmeans.__dict__["n_features_in_"] = checkpoint["n_features_in_"]
        kmeans.__dict__["_n_threads"] = checkpoint["_n_threads"]
        kmeans.__dict__["cluster_centers_"] = checkpoint["cluster_centers_"].numpy()
    return kmeans


def kmeans100(pretrained: bool = True) -> KMeans:
    r"""
    k-means checkpoint for HuBERT-Discrete with 100 clusters.
    Args:
        pretrained (bool): load pretrained weights into the model
        progress (bool): show progress bar when downloading model
    """
    return _kmeans(100, pretrained)

'''code for extracting discrete features'''
# hubert_weight = torch.load('/home/cottonlove/hubert/checkpoints/hubert-discrete-96b248c5.pt')
kmeans = kmeans100(pretrained=True)
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
# hubert_weight = torch.load('/home/cottonlove/hubert/checkpoints/hubert-discrete-96b248c5.pt')
# hubert = HubertSSL()
# consume_prefix_in_state_dict_if_present(hubert_weight["hubert"], "module.") #To let a non-DDP model load a state dict from a DDP model, consume_prefix_in_state_dict_if_present() needs to be applied to strip the prefix “module.” in the DDP state dict before loading.
# hubert.load_state_dict(hubert_weight["hubert"])
# hubert.eval()
# hubert = hubert.cuda() #cuda로 안올려주면 input이랑 모델 weight가 같은 type 아니라고 error 뜬다.

# #print("type of hubert: ", hubert)

# # # # Load audio
# source, sr = torchaudio.load("/home/cottonlove/baseline_code/DB/LibriTTS/train-clean-100/19/198/19_198_000000_000002.wav")
# source = torchaudio.functional.resample(source, sr, 16000)
# source = source.unsqueeze(0).cuda()

# # print("source shape: ", source.shape) #source shape:  torch.Size([1, 1, 84800])


# SSLfeatures = hubert.SSLfeatures(source)

# print("SSLfeatures.shape: ", SSLfeatures)
# torch.save(SSLfeatures, 'softvc_SSLfeatures_6.pt')


###### cached 된거랑 내가 만든거랑 왜 차이가 나죠
# Reload tensors from files
ssl_layer7 = torch.load('softvc_SSLfeatures_7.pt').to('cuda:0') # 밑에 torch.equal로 비교할때 다른 device에 있음 에러남
print("ssl_layer7.shape: ",ssl_layer7.shape) #torch.Size([1, 265, 768]
ssl_layer6 = torch.load('softvc_SSLfeatures_6.pt').to('cuda:0')
print("ssl_layer6.shape: ",ssl_layer6.shape) #torch.Size([1, 265, 768])
style_ssl_layer6 = torch.load('/home/cottonlove/baseline_code/stylebook/stylebook_layer6_ssl.pt').to('cuda:0')
style_ssl_layer6_woPAD = torch.load('/home/cottonlove/baseline_code/stylebook/stylebook_layer6_ssl_woPAD.pt').to('cuda:0')


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