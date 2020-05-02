import os
import numpy as np
import cv2
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Sets random seeds and some other stuff for reproducibility
torch.manual_seed(42)  # Note that this does not always ensure reproducible results
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

if "train" not in os.listdir():
    os.system("wget https://storage.googleapis.com/exam-deep-learning/train-Exam2.zip")
    os.system("unzip train-Exam2.zip")

DATA_DIR = os.getcwd() + "/train/"
RESIZE_TO = 224
def cropped(path):
    image = cv2.imread(path)
    # percent by which the image is resized
    scale_percent = RESIZE_TO/image.shape[0]
    # calculate the 50 percent of original dimensions
    width = int(image.shape[1] * scale_percent)
    height = RESIZE_TO
    # dsize
    dsize = (width, height)
    # resize image
    output = cv2.resize(image, dsize)
    x = int((width-RESIZE_TO) / 2)
    w = RESIZE_TO
    crop = output[:, x:x + w]
    return crop

def pairing(img1, img2):
    img1[:, :, 0] = (img1[:, :, 0] + img2[:, :, 0]) / 2
    img1[:, :, 1] = (img1[:, :, 1] + img2[:, :, 1]) / 2
    img1[:, :, 2] = (img1[:, :, 2] + img2[:, :, 2]) / 2
    return img1

x, y = [], []
for path in [f for f in os.listdir(DATA_DIR) if f[-5:] == ".json"]:
    with open(DATA_DIR + path[:-5] + ".txt", "r") as s:
        label = []
        for line in s:
            text = line.strip()
            if text != "red blood cell":
                label.append(text)
    y.append(label)
    img = cv2.cvtColor(cropped(DATA_DIR + path[:-5] + ".png"), cv2.COLOR_BGR2HSV)
    x.append(img)
for i in range(len(x)):
    paired = x[np.random.randint(len(x))]
    x.append(pairing(x[i], paired))
    paired = x[np.random.randint(len(x))]
    x.append(pairing(x[i], paired))
    y.append(y[i])
    y.append(y[i])

mlb = MultiLabelBinarizer()
mlb.fit([["difficult", "gametocyte", "trophozoite", "ring", "schizont", "leukocyte"]])
# 929 255 101 462 204 125 59
# 988 269 107 510 216 128 118
# 1089 304 208 568 234 136 124
# 1332 367 228 701 263 267 245
# 1433 402 329 759 281 275 251
# 1492 416 335 807 293 278 310
#      451 343 844 298 403 313
#      485 361 932 502 408 325
#      499 367 980 514 411 384
#      534 468 1038 532 419 390
#      548 474 1086 544 422 449
#      583 482 1123 549 547 452
#      597 488 1171 561 550 511
#      632 589 1229 579 558 517
#      646 595 1277 591 561 576
#      681 603 1314 596 686 579
#      695 609 1362 608 689 638
#      729 627 1450 812 694 650
#      764 728 1508 830 702 656
#      778 734 1556 842 705 715
#      813 742 1593 847 830 718
y = mlb.transform(y)
# img.shape (1200, 1600, 3)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, stratify=y)
x_train, x_test, y_train, y_test = torch.FloatTensor(x_train).to(device), torch.FloatTensor(x_test).to(device), \
                                   torch.FloatTensor(y_train).to(device), torch.FloatTensor(y_test).to(device)

x_train[:, :, :, 0] = (x_train[:, :, :, 0] - 85.711) / 46.7
x_train[:, :, :, 1] = (x_train[:, :, :, 1] - 31.639) / 44.6
x_train[:, :, :, 2] = (x_train[:, :, :, 2] - 197.631) / 52.3
x_test[:, :, :, 0] = (x_test[:, :, :, 0] - 85.711) / 46.7
x_test[:, :, :, 1] = (x_test[:, :, :, 1] - 31.639) / 44.6
x_test[:, :, :, 2] = (x_test[:, :, :, 2] - 197.631) / 52.3

x_train.requires_grad = True
x_train = x_train.permute(0, 3, 1, 2).contiguous()
x_test = x_test.permute(0, 3, 1, 2)
x, y  = None, None

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=6):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.bt = BasicConv2d(2048, num_classes, kernel_size=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.bt(x)
        # 8 x 8 7
        x = torch.mean(x.view(x.size(0), x.size(1), -1), dim=2)

        return x

def resnet50(**kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model

lmbda = lambda epoch: 0.99
N_EPOCHS = 1000
BATCH_SIZE = 64

model = resnet50().to(device)
model.load_state_dict(torch.load("model_xinyu_yao.pt"))
optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lmbda)
criterion_train = nn.BCEWithLogitsLoss(reduce=True, size_average=False)
criterion_test = nn.BCELoss(reduce=True, size_average=False)

# %% -------------------------------------- Training Loop ----------------------------------------------------------
loss_test_best = 3.69370
m = nn.Sigmoid()
print("Starting training loop...")
for epoch in range(N_EPOCHS):
    loss_train = 0
    model.train()
    for batch in range(len(x_train)//BATCH_SIZE + 1):
        inds = slice(batch*BATCH_SIZE, (batch+1)*BATCH_SIZE)
        optimizer.zero_grad()
        p = x_train[inds]
        logits = model(p)
        loss = criterion_train(logits, y_train[inds])
        loss.backward()
        optimizer.step()
        loss_train += loss.item()

    model.eval()
    with torch.no_grad():
        y_test_pred = model(x_test)
        loss = criterion_test(m(y_test_pred), y_test)
        loss_test = loss.item()/len(x_test)

    print("Epoch {} | Train Loss {:.5f} - Test Loss {:.5f}".format(epoch, loss_train/len(x_train), loss_test))

    if loss_test < loss_test_best:
        torch.save(model.state_dict(), "model_xinyu_yao.pt")
        print("The model has been saved!")
        loss_test_best = loss_test
    scheduler.step()

# %% ------------------------------------------ Final test -------------------------------------------------------------
model.load_state_dict(torch.load("model_xinyu_yao.pt"))
model.eval()
with torch.no_grad():
    y_test_pred = model(x_test.permute(0, 3, 1, 2))
    loss = criterion_test(m(y_test_pred), y_test)
    loss_test = loss.item()
print("The score on the test set is {:.2f}".format(loss_test))
# %% -------------------------------------------------------------------------------------------------------------------