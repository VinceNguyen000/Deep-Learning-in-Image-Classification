import matplotlib.pyplot as plt
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from evaluation import compute_confusion_matrix
from plotting import plot_accuracy, plot_confusion_matrix
from train import train_model
from scipy.io import savemat


# Build VGG model
class VGG(torch.nn.Module):
    def __init__(self, num_classes, ):
        super(VGG, self).__init__()
        self.num_classes = num_classes
        # convolutional layers
        self.conv_layers = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(128, 256, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(256, 512, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(512, 512, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(512, 512, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(512, 512, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )

        height, width = 7, 7
        self.avgpool = torch.nn.AdaptiveAvgPool2d((height, width))
        self.feature_extractor = torch.nn.Linear(512 * height * width, 4096)
        # fully connected linear layers
        self.linear_layers = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(in_features=4096, out_features=4096),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(in_features=4096, out_features=self.num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.avgpool(x)
        # flatten to prepare for the fully connected layers
        x = x.view(x.size(0), -1)
        feature = self.feature_extractor(x)
        prob = self.linear_layers(feature)

        return prob, feature


sumWriter = SummaryWriter()

# Initialize batch_size and number of epoch
BATCH_SIZE = 16
NUM_EPOCHS = 3
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Initialize the path of our data set
datasets_path = 'data2/NWPURESISC45'

# Transform the dataset
datasets_transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.RandomRotation(10),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Build a dataset from the  path of our data set
datasets = torchvision.datasets.ImageFolder(root=datasets_path, transform=datasets_transforms)

# Divide our dataset to train set, valid set, and test set that we will use in this exercise
train_set, valid_set, test_set = torch.utils.data.random_split(datasets, [5000, 1000, 4500])

# Build data loader
train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=True)

# Build the model
model = VGG(num_classes=15)
# print(model)
# print(model.conv_layers)
# print(model.conv_layers[18])
# fcLayer = model.conv_layers[18]
# print(fcLayer)
# featuresModel = nn.Sequential(*list(model.features.children())[:-1])
# aFeature = model.feature_extractor[:5](input)
# print(output)
model = model.to(DEVICE)
optimizer = torch.optim.SGD(model.parameters(), momentum=0.9, lr=0.01)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                       factor=0.1,
                                                       mode='max',
                                                       verbose=True)
#part 1
featureList = []
labelsList = []
for i, (aImage, aLabel) in enumerate(test_loader):
    # print(aLabel)
    aImage = aImage.to(DEVICE)
    aLabel = aLabel.to(DEVICE)
    _, dataFeature = model(aImage)
    featureEachImage = dataFeature.cpu().detach().numpy()
    featureList.append(featureEachImage)
    labeEachImage = aLabel.cpu().detach().numpy()
    labelsList.append(labeEachImage)
savemat("matlab_features.mat", {"data": featureList})
savemat("matlab_labels.mat", {"data": labelsList})
#print(len(featureList))
#print(featureList)
#part 2
model.load_state_dict(torch.load('model-vgg-final-55-76.ckpt',
                                 map_location=torch.device('cpu')))

batch_loss_list, train_accuracy_list, valid_accuracy_list = train_model(
    model=model,
    num_epochs=NUM_EPOCHS,
    train_loader=train_loader,
    valid_loader=valid_loader,
    test_loader=test_loader,
    device=DEVICE,
    scheduler=scheduler,
    scheduler_on='valid_acc')
sumWriter.close()
plot_accuracy(train_accuracy_list=train_accuracy_list,
              valid_accuracy_list=valid_accuracy_list,
              results_dir=None)
plt.ylim([60, 100])
plt.savefig('accuracy.png')
plt.show()
class_dict = {0: 'airplane',
              1: 'airport',
              2: 'baseball_diamond',
              3: 'basketball_court',
              4: 'beach',
              5: 'bridge',
              6: 'chaparral',
              7: 'church',
              8: 'circular_farmland',
              9: 'cloud',
              10: 'commercial_area',
              11: 'dense_residential',
              12: 'desert',
              13: 'forest',
              14: 'freeway'}
mat = compute_confusion_matrix(model=model, data_loader=test_loader, device=torch.device('cpu'))
plot_confusion_matrix(mat, class_names=class_dict.values())
plt.show()
