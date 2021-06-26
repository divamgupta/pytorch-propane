# Pytorch Propane 

Pytorch Propane is a simplified wrapper to make training and evaluation of neural networks easy and scalable. 

### Features 

* Keras like functions ( e.g. .fit() ) to train your models 
* Automatic command line mode - Don't spend time writing command line interfaces. 
* Flexible API - which empowers you, not get in the way. 



Example :

```python
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


model = Model(network=Net())

model.compile( optimizer='adam'  )
model.add_loss( "nll_loss" , display_name="nll_losss" )

model.train_step(data_x=torch.zeros((3,1,28,28)) , data_y=torch.zeros((3)).long())
```

