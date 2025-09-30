from data import PhysioNet2012, PhysioNet2019, UEA
from torch.utils.data import DataLoader

physioNet2012 = PhysioNet2012(split="train", train_prop=0.7)
data_loader_1 = DataLoader(physioNet2012, batch_size=32, shuffle=True)

physioNet2019 = PhysioNet2019(split="train", train_prop=0.7)
data_loader_2 = DataLoader(physioNet2019, batch_size=32, shuffle=True)

uea = UEA("BasicMotions", split="train", train_prop=0.7)
data_loader_3 = DataLoader(uea, batch_size=32, shuffle=True)
