import torch

device = torch.device(
    "cuda" if torch.cuda.is_available() 
    else "cpu"
)



if device == torch.device("cuda"):
    print(f"😁😁😎 Using device: {device}")
else:
    print(f"😡😡🫨  Using device: {device}")