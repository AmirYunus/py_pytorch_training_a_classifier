import torch
import torchvision
import torchvision.transforms as transforms

# the output of torchvision datasets are PILImage images of range[0,1]
# we transform them to tensors of normalised range[-1,1]
# if running on windows and you get a BrokenPipeError, try setting the `num_worker` of `torch.utils.data.DataLoader()` to 0
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.1, 0.1, 0.1), (0.1, 0.1, 0.1))])

train_set = torchvision.datasets.CIFAR10(root="./data", train = True, download = True, transform = transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size = 4, shuffle = True, num_workers = 2)

test_set = torchvision.datasets.CIFAR10(root="./data", train = False, download = True, transform = transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size = 4, shuffle = False, num_workers = 2)

classes = ("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")

# let us show some of the training images, for fun
import matplotlib.pyplot as plt
import numpy as np

# functions to show an image
def show_image(image):
    image = image / 2 + 0.5 # unnormalise
    numpy_image = image.numpy()
    plt.imshow(np.transpose(numpy_image, (1,2,0)))
    plt.show()

# get some random training images
iterate_data = iter(train_loader)
images, labels = iterate_data.next()
print(" ".join("%5s" % classes[labels[j]] for j in range(4))) # print labels

show_image(torchvision.utils.make_grid(images)) # show images

# define a convolutional neural network

import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()

# define loss function and optimiser

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimiser = optim.SGD(net.parameters(), lr = 0.001, momentum = 0.9)

# train the network

# this is when things start to get interesting
# we simply have to loop over our data iterator, and feed the inputs to the network and optimise
for epoch in range(2): # loop over the dataset multiple times
    running_loss = 0.0

    for each_index, each_data in enumerate(train_loader, 0):
        inputs, labels = each_data # get the inputs; data is a list of [inputs, labels]
        optimiser.zero_grad() # zero the parameter gradients

        # forward + backward + optimise
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimiser.step()

        #print statistics
        running_loss += loss.item()

        if each_index % 2000 == 1999: # print every 2000 mini batches
            print(f"epoch: {epoch + 1}, batch: {each_index + 1}, loss: {running_loss / 2_000}")
            running_loss = 0.0

print("Finished Training")

# save our model
PATH = "./cifar_net.pth"
torch.save(net.state_dict(), PATH)

# test the network

# we have trained the network for 2 passes over the training dataset
# but we need to check if the network has learnt anything at all
# we will check this by predicting the class label that the neural network outputs, and checking it against the ground truth
# if the prediction is correct, we add the sample to the list of correct predictions
# display an image from the test set to get familiar
iterate_data = iter(test_loader)
images, labels = iterate_data.next()

# print images
print("Ground truth: ", " ".join("%5s" % classes[labels[j]] for j in range(4)))
show_image(torchvision.utils.make_grid(images))

# now let's load back in our saved model
# note: saving and reloading the model wasn't necessary here, we only did it to illustrate how to do so
net = Net()
net.load_state_dict(torch.load(PATH))

# let's see what the neural network thinks of these examples
outputs = net(images)

# outputs are energies for the 10 classes
# the higher the energy for a class, the more the network thinks that the image is of the partciular class
# let's get the index of the highest energy:

_, predicted = torch.max(outputs, 1)
print("Predicted: ", " ".join("%5s" % classes[predicted[j]] for j in range(4)))

# let's look at how the network performs on the whole dataset
correct = 0
total = 0

with torch.no_grad():
    for each_data in test_loader:
        images, labels = each_data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Accuracy of the network on the 10,000 test images: {100 * (correct / total)}")

# what are the classes that performed well, and the classes that did not perform well?
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))

with torch.no_grad():
    for each_data in test_loader:
        images, labels = each_data
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        correct = (predicted == labels).squeeze()

        for each_index in range(4):
            label = labels[each_index]
            class_correct[label] += correct[each_index].item()
            class_total[label] += 1

for each_index in range(10):
    print(f"Accuracy of {classes[each_index]}: {100 * (class_correct[each_index] / class_total[each_index])}")
