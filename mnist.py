import torch
import torchvision
from utils import plot_image


def load_mnist_dataset(batch_size = 512, shuffle=True):
    # mnist dataset
    mnist_train_dataset = torchvision.datasets.MNIST('mnist_data', train=True, download=True,
                                transform=torchvision.transforms.Compose([
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize(
                                        (0.1307,), (0.3081,))
                                ]))
    
    mnist_train_dataset, mnist_validation_dataset = torch.utils.data.random_split(mnist_train_dataset, [0.85, 0.15])
    print(len(mnist_train_dataset), len(mnist_validation_dataset))
    mnist_train_dataloader = torch.utils.data.DataLoader(mnist_train_dataset, batch_size=batch_size, shuffle=shuffle)
    mnist_validation_dataloader = torch.utils.data.DataLoader(mnist_validation_dataset, batch_size=batch_size, shuffle=shuffle)

    mnist_test_dataset = torchvision.datasets.MNIST('mnist_data/', train=False, download=True,
                                transform=torchvision.transforms.Compose([
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize(
                                        (0.1307,), (0.3081,))
                                ]))
    mnist_test_loader = torch.utils.data.DataLoader(mnist_test_dataset, batch_size=batch_size, shuffle=shuffle)

    return mnist_train_dataloader, mnist_validation_dataloader, mnist_test_loader


if __name__ == "__main__":

    # mnist dataset
    mnist_train_dataloader, mnist_validation_dataloader, _ = load_mnist_dataset()

    # 展示一个batch的数据
    x, y = next(iter(mnist_train_dataloader))
    print(x.shape, y.shape, x.min(), x.max())
    plot_image(x, y, 'image sample')