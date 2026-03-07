import torch
print(torch.cuda.is_available())   
print(torch.cuda.device_count())   
print(torch.cuda.get_device_name(0))   
if torch.cuda.is_available():
    x = torch.randn(3, 3).to('cuda')
    print(x.device)  # Should output: cuda:0   
print(torch.version.cuda)         # PyTorch's CUDA version
print(torch.backends.cudnn.enabled) # cuDNN is enabled
print(torch.backends.cudnn.version()) # cuDNN version   