from model import Net
from config import VinConfig
from train import Trainer


def main():
    net = Net(VinConfig)
    net = net.cuda()
    net = net.double()
    trainer = Trainer(VinConfig, net)
    trainer.train()


if __name__ == '__main__':
    main()
