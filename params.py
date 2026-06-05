import torch
from src.Vae import Vae

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Vae().to(device)
    model.report_parameters()
