import torch
from torchvision.transforms import ToPILImage

from GAN.model import Generator


def generate(device, model):
    model.eval()
    output = model.sample(device)
    output = output[0].detach().cpu()
    img = ToPILImage()(output)
    img.save("generated.jpg")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = Generator().to(device)
    generator.load_state_dict(torch.load("./demo_model.pth"))
    generate(device=device, model=generator)
