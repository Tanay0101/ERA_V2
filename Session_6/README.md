```
class MNISTSmallConv(nn.Module):
    def __init__(self):
      super(MNISTSmallConv, self).__init__()
      # n = 13
      self.conv1 = nn.Conv2d(1, 16, 3, stride=2) #input -? OUtput? RF
      # n = 11
      self.conv2 = nn.Conv2d(16, 32, 3)
      # n = 5
      self.pool1 = nn.MaxPool2d(2, 2)
      # n = 3
      self.conv3 = nn.Conv2d(32, 32, 3)
      self.linear1 = nn.Linear(288, 16)
      self.linear2 = nn.Linear(16, 10)

      self.drop1 = nn.Dropout(0.1)
      self.drop2 = nn.Dropout(0.1)
      self.drop3 = nn.Dropout(0.1)
      self.drop4 = nn.Dropout(0.1)

      self.batchnorm1 = nn.BatchNorm2d(16)
      self.batchnorm2 = nn.BatchNorm2d(32)
      self.batchnorm3 = nn.BatchNorm2d(32)
      self.batchnorm4 = nn.BatchNorm1d(16)

    def forward(self, x):
      x = self.conv1(x)
      x = self.batchnorm1(x)
      x = F.relu(x)
      x = self.drop1(x)

      x = self.conv2(x)
      x = self.batchnorm2(x)
      x = F.relu(x)
      x = self.pool1(x)
      x = self.drop2(x)

      x = self.conv3(x)
      x = self.batchnorm3(x)
      x = F.relu(x)
      x = self.drop3(x)

      x = x.view(-1, 288)

      x = self.linear1(x)
      x = self.batchnorm4(x)
      x = F.relu(x)

      x = self.linear2(x)
      return F.log_softmax(x, dim=1)

```

```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 16, 13, 13]             160
       BatchNorm2d-2           [-1, 16, 13, 13]              32
           Dropout-3           [-1, 16, 13, 13]               0
            Conv2d-4           [-1, 32, 11, 11]           4,640
       BatchNorm2d-5           [-1, 32, 11, 11]              64
         MaxPool2d-6             [-1, 32, 5, 5]               0
           Dropout-7             [-1, 32, 5, 5]               0
            Conv2d-8             [-1, 32, 3, 3]           9,248
       BatchNorm2d-9             [-1, 32, 3, 3]              64
          Dropout-10             [-1, 32, 3, 3]               0
           Linear-11                   [-1, 16]           4,624
      BatchNorm1d-12                   [-1, 16]              32
           Linear-13                   [-1, 10]             170
================================================================
Total params: 19,034
Trainable params: 19,034
Non-trainable params: 0
----------------------------------------------------------------
```

The model architecture incorporates a blend of convolutional and linear layers. The initial convolutional layer employs a stride of 2 to downsize the image. This choice is made under the presumption that this reduction won't significantly compromise information loss. Following the convolutional layers, batch normalization layers are applied to enhance gradient stability and expedite training. Dropout is utilized as a regularization technique. The final layers are linear to transform the extracted features to the desired output format. 