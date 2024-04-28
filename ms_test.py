import mindspore
from mindspore import nn, Tensor
from mindspore.dataset import MnistDataset
from mindspore.dataset.transforms.c_transforms import Compose, OneHot, ToTensor
from mindspore.nn.probability.dpn import NormalPrior
from mindspore.train import Model


# 定义CVAE模型
class CVAE(nn.Cell):
    def __init__(self, latent_dim, num_classes):
        super(CVAE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.SequentialCell([
            nn.Dense(784, 512),
            nn.ReLU(),
            nn.Dense(512, latent_dim * 2)  # 输出均值和对数方差
        ])
        self.decoder = nn.SequentialCell([
            nn.Dense(latent_dim + num_classes, 512),
            nn.ReLU(),
            nn.Dense(512, 784),
            nn.Sigmoid()  # 输出像素值在[0, 1]之间
        ])
        self.normal_prior = NormalPrior()

    def construct(self, x, labels):
        # 编码
        mu, log_var = self.encoder(x).split(2, 1)
        z = self.normal_prior.sample((x.shape[0], self.latent_dim), mu, log_var)
        # 解码
        z_labels = mindspore.ops.Concat(1)([z, labels])
        recon_x = self.decoder(z_labels)
        return recon_x, mu, log_var

# 定义损失函数
class CVAELoss(nn.Cell):
    def __init__(self, beta=1.0):
        super(CVAELoss, self).__init__()
        self.recon_loss = nn.BCELoss(reduction='sum')
        self.kl_loss = nn.KLDivergence(reduction='sum')
        self.beta = beta

    def construct(self, recon_x, x, mu, log_var):
        # 重构损失
        recon_loss = self.recon_loss(recon_x, x)
        # KL散度损失
        kl_loss = self.kl_loss(mu, log_var)
        # 总损失
        total_loss = recon_loss + self.beta * kl_loss
        return total_loss

# 数据集准备
def create_dataset(batch_size=32):
    mnist_ds = MnistDataset("/path/to/mnist")
    transform = Compose([ToTensor(), OneHot(num_classes=10)])
    mnist_ds = mnist_ds.map(operations=transform, input_columns=["image", "label"])
    mnist_ds = mnist_ds.batch(batch_size)
    return mnist_ds

# 超参数设置
latent_dim = 2
num_classes = 10
num_epochs = 10
batch_size = 64
learning_rate = 1e-3

# 模型初始化
cvae = CVAE(latent_dim, num_classes)
loss_fn = CVAELoss(beta=1.0)
optimizer = nn.Adam(cvae.trainable_params(), learning_rate)

# 训练模型
train_dataset = create_dataset(batch_size)
model = Model(cvae, loss_fn, optimizer)
model.train(num_epochs, train_dataset)

# 生成新样本
def generate_samples(model, num_samples, num_classes):
    z = Tensor(mindspore.numpy.random.randn(num_samples, latent_dim), mindspore.float32)
    labels = Tensor(mindspore.numpy.eye(num_classes)[None, :], mindspore.float32)
    samples = model.decoder(mindspore.ops.Concat(1)([z, labels]))
    return samples

# 生成样本
generated_samples = generate_samples(cvae, 10, num_classes)
