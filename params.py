import torch
from src.Vae import DualBranchPointVAE

if __name__ == "__main__":
    # Teste rápido para verificar se o modelo compila e roda um forward pass
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DualBranchPointVAE().to(device)
    model.report_parameters()
