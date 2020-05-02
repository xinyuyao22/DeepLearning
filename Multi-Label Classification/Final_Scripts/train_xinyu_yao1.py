import os
import numpy as np
import cv2
import json
from sklearn.preprocessing import MultiLabelBinarizer
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Sets random seeds and some other stuff for reproducibility
torch.manual_seed(42)  # Note that this does not always ensure reproducible results
#np.random.seed(42)  # (See https://pytorch.org/docs/stable/notes/randomness.html)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

if "train" not in os.listdir():
    os.system("wget https://storage.googleapis.com/exam-deep-learning/train-Exam2.zip")
    os.system("unzip train-Exam2.zip")
DATA_DIR = os.getcwd() + "/"

with open(DATA_DIR + "test_file.txt", "r") as s:
    test_path = []
    for line in s:
        text = line.strip()
        test_path.append(text)

DATA_DIR = os.getcwd() + "/train/"
RESIZE_TO = 224
def cropped(image):
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
def noised(img):
    mean = 0
    var = 10
    sigma = var ** 0.5
    gaussian = np.random.normal(mean, sigma, (224, 224))  # np.zeros((224, 224), np.float32)
    noisy_image = np.zeros(img.shape, np.float32)
    noisy_image[:, :, 0] = img[:, :, 0] + gaussian
    noisy_image[:, :, 1] = img[:, :, 1] + gaussian
    noisy_image[:, :, 2] = img[:, :, 2] + gaussian
    return noisy_image
def rand_proc(img):
    rand_proc = [cv2.flip(img, 0), cv2.flip(img, 1), cv2.flip(img, -1), cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE),
                 cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE), cv2.GaussianBlur(img, (5, 5), 0)]
    i = np.random.randint(6)
    return rand_proc[i]
x_train, x_test, y_train, y_test = [], [], [], []
for path in [f for f in os.listdir(DATA_DIR) if f[-4:] == ".txt"]:
    flag = False
    with open(DATA_DIR + path, "r") as s:
        label = []
        for line in s:
            text = line.strip()
            if text != "red blood cell":
                flag = True
                label.append(text)
    img = cv2.imread(DATA_DIR + path[:-4] + ".png")
    crop_img = cv2.cvtColor(cropped(img), cv2.COLOR_BGR2HSV)
    if path in test_path:
        y_test.append(label)
        x_test.append(crop_img)
    else:
        y_train.append(label)
        x_train.append(crop_img)
    if flag == True:
        with open(DATA_DIR + path[:-4] + ".json") as f:
            annos = json.load(f)
        re_boxes, d_boxes, g_boxes, t_boxes, ri_boxes, s_boxes, l_boxes = [], [], [], [], [], [], []
        for ix, ann in enumerate(annos):
            bbox = ann['bounding_box']
            minr = bbox['minimum']['r']
            minc = bbox['minimum']['c']
            maxr = bbox['maximum']['r']
            maxc = bbox['maximum']['c']
            cls = ann['category']
            if cls == 'red blood cell':
                re_boxes.append([minr, minc, maxr, maxc])
            elif cls == 'difficult':
                d_boxes.append([minr, minc, maxr, maxc])
            elif cls == 'gametocyte':
                g_boxes.append([minr, minc, maxr, maxc])
            elif cls == 'trophozoite':
                t_boxes.append([minr, minc, maxr, maxc])
            elif cls == 'ring':
                ri_boxes.append([minr, minc, maxr, maxc])
            elif cls == 'schizont':
                s_boxes.append([minr, minc, maxr, maxc])
            else:
                l_boxes.append([minr, minc, maxr, maxc])
            for i in range(int(len(re_boxes) / 10)):
                for t in t_boxes:
                    rc = np.random.randint(len(re_boxes))
                    resized = cv2.resize(rand_proc(img[t[0]:t[2], t[1]:t[3], :]), (re_boxes[rc][3] - re_boxes[rc][1], re_boxes[rc][2] - re_boxes[rc][0]))
                    img[re_boxes[rc][0]:re_boxes[rc][2], re_boxes[rc][1]:re_boxes[rc][3], :] = resized
            for i in range(int(len(re_boxes) / 9)):
                for g in g_boxes:
                    rc = np.random.randint(len(re_boxes))
                    resized = cv2.resize(rand_proc(img[g[0]:g[2], g[1]:g[3], :]), (re_boxes[rc][3] - re_boxes[rc][1], re_boxes[rc][2] - re_boxes[rc][0]))
                    img[re_boxes[rc][0]:re_boxes[rc][2], re_boxes[rc][1]:re_boxes[rc][3], :] = resized
            for i in range(int(len(re_boxes) / 7)):
                for ri in ri_boxes:
                    rc = np.random.randint(len(re_boxes))
                    resized = cv2.resize(rand_proc(img[ri[0]:ri[2], ri[1]:ri[3], :]), (re_boxes[rc][3] - re_boxes[rc][1], re_boxes[rc][2] - re_boxes[rc][0]))
                    img[re_boxes[rc][0]:re_boxes[rc][2], re_boxes[rc][1]:re_boxes[rc][3], :] = resized
            for i in range(int(len(re_boxes) / 6)):
                for d in d_boxes:
                    patch = rand_proc(img[d[0]:d[2], d[1]:d[3], :])
                    r = np.random.randint(img.shape[0] - patch.shape[0])
                    c = np.random.randint(img.shape[1] - patch.shape[1])
                    img[r:(r + patch.shape[0]), c:(c + patch.shape[1]), :] = patch
            for i in range(int(len(re_boxes) / 5)):
                for s in s_boxes:
                    patch = rand_proc(img[s[0]:s[2], s[1]:s[3], :])
                    r = np.random.randint(img.shape[0] - patch.shape[0])
                    c = np.random.randint(img.shape[1] - patch.shape[1])
                    img[r:(r + patch.shape[0]), c:(c + patch.shape[1]), :] = patch
            for i in range(int(len(re_boxes) / 4)):
                for l in l_boxes:
                    patch = rand_proc(img[l[0]:l[2], l[1]:l[3], :])
                    r = np.random.randint(img.shape[0] - patch.shape[0])
                    c = np.random.randint(img.shape[1] - patch.shape[1])
                    img[r:(r + patch.shape[0]), c:(c + patch.shape[1]), :] = patch
    crop_img = cv2.cvtColor(cropped(img), cv2.COLOR_BGR2HSV)
    x_train.append(crop_img)
    y_train.append(label)
    if "leukocyte" in label:
        y_train.append(label)
        y_train.append(label)
        y_train.append(label)
        y_train.append(label)
        y_train.append(label)
        y_train.append(label)
        y_train.append(label)
        y_train.append(label)
        y_train.append(label)
        y_train.append(label)
        x_train.append(noised(crop_img))
        im = np.copy(crop_img)
        im[:, :, 0] = crop_img[:, :, 0] + 5
        x_train.append(im)
        im[:, :, 0] = crop_img[:, :, 0] - 5
        x_train.append(im)
        im[:, :, 0] = crop_img[:, :, 0] + 10
        x_train.append(im)
        im[:, :, 0] = crop_img[:, :, 0] - 10
        x_train.append(im)
        im[:, :, 1] = crop_img[:, :, 1] + 5
        x_train.append(im)
        im[:, :, 1] = crop_img[:, :, 1] + 10
        x_train.append(im)
        im[:, :, 1] = crop_img[:, :, 1] + 15
        x_train.append(im)
        im[:, :, 1] = crop_img[:, :, 1] + 20
        x_train.append(im)
        im[:, :, 2] = crop_img[:, :, 2] - 20
        x_train.append(im)
    if "schizont" in label:
        y_train.append(label)
        y_train.append(label)
        y_train.append(label)
        y_train.append(label)
        x_train.append(noised(crop_img))
        im = np.copy(crop_img)
        im[:, :, 2] = crop_img[:, :, 2] - 5
        x_train.append(im)
        im[:, :, 2] = crop_img[:, :, 2] - 10
        x_train.append(im)
        im[:, :, 2] = crop_img[:, :, 2] - 15
        x_train.append(im)


mlb = MultiLabelBinarizer()
mlb.fit([["difficult", "gametocyte", "trophozoite", "ring", "schizont", "leukocyte"]])

y_train = mlb.transform(y_train)
y_test = mlb.transform(y_test)
# img.shape (1200, 1600, 3)
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
ann, annos, bbox, cls, crop_img, d, d_boxes, flag, g, g_boxes, i, img, ix, l, l_boxes, label, line, minr, maxr, minc, maxc, path, rc, re_boxes, resized, ri, ri_boxes, s_boxes, t, t_boxes, test_path, text = None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None
LR = 0.0001
lmbda = lambda epoch: 0.99
N_EPOCHS = 1000
BATCH_SIZE = 64

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

model = resnet50().to(device)
model.load_state_dict(torch.load("model_xinyu_yao.pt"))
optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lmbda)
criterion_train = nn.BCEWithLogitsLoss(reduce=True, size_average=False)
criterion_test = nn.BCELoss(reduce=True, size_average=False)

# %% -------------------------------------- Training Loop ----------------------------------------------------------
loss_test_best = 3.45882
m = nn.Sigmoid()
print("Starting training loop...")
for epoch in range(N_EPOCHS):
    loss_train = 0
    model.train()
    for batch in range(len(x_train)//BATCH_SIZE + 1):
        inds = slice(batch*BATCH_SIZE, (batch+1)*BATCH_SIZE)
        optimizer.zero_grad()
        logits = model(x_train[inds])
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