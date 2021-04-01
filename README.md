# Machine-Learning

Repository for Machine Learning models implemententation for different python libraries

Requirements 
- torch


###### Torch examples  

##### Simple neural net

```
NeuralNet(100, [[15,True],[20,False],2],[[torch.nn.ReLU(),torch.nn.BatchNorm1d(15)],
                                        [torch.nn.ReLU(), torch.nn.BatchNorm1d(20)],
                                        torch.nn.Sigmoid()])
```

- 100 -> input_shape
- [[15,True],[20,False],2] -> 3 layers  
--- [15,True] -> First layer nn.Linear(100,15,bias=True)  
--- [20,False] -> Second layer nn.Linear(15,20,bias=False)  
--- 2 -> Last layer nn.Linear(20,2) -> bias true by default  

#### ConvNet
# How to create ConvNets
```
model = ConvNet(3 , [['conv2d', '15,3,2'],
                    ['conv2dt', '35,3,2'],
                    ['conv2dt', '55,3,2'],
                    ['conv2d', '50,3,2']],
                     [[nn.MaxPool2d((2,2)),nn.BatchNorm2d(15),nn.ReLU()],  
                      'skip', 
                      nn.ReLU(),  
                      nn.ReLU()])  
print(model(torch.randn(1,3,100,100)).shape)  
```
Or
```
model = ConvNet(3 # input_channels, 
                    [['conv2d', 'out_channels=15,kernel_size=3,stride=2'], # Conv2d
                    ['conv2dt', 'out_channels=30,kernel_size=3,stride=2'], # Conv2Transpose
                    ['conv2dt', 'out_channels=50,kernel_size=3,stride=2'],
                    ['conv2d', 'out_channels=55,kernel_size=3,stride=2,padding=(2,2)']],
                    [[nn.MaxPool2d((2,2)),nn.BatchNorm2d(15),nn.ReLU()],
                     'skip', 
                     nn.ReLU(),
                     nn.ReLU()])
```

Both models are the same written in different parametric way



For creating models we need to use the CustomModel class which can combine for us any layers defined by us.  
```
from models.utils import ImageClass

model = ConvNet(3, [['conv2d', 'out_channels=50,kernel_size=15,stride=2'],
                    ['conv2d', 'out_channels=100,kernel_size=7,stride=2'],
                    ['conv2d', 'out_channels=150,kernel_size=3,stride=1'],
                    ['conv2d', 'out_channels=100,kernel_size=3,stride=1']],
                    [[nn.MaxPool2d((2,3)),nn.BatchNorm2d(50),nn.ReLU()],
                     [nn.MaxPool2d((2,2)),nn.BatchNorm2d(100),nn.ReLU()], 
                     [nn.MaxPool2d((2,2)),nn.BatchNorm2d(150),nn.ReLU()],
                     [nn.ReLU(),nn.Flatten()]])

with torch.no_grad():
    model.eval()
    out_shape = model(torch.randn((1,3,250,250),requires_grad=False)).shape[1]
    model.train()
print(out_shape)
# NeuralNet for feature classification extracted by the first model
model2 = NeuralNet(out_shape, [528,500,9],
              [torch.nn.ReLU(),torch.nn.ReLU(),"skip"])

# We put ConvNet and Neural net in a list to pass to our custom model which combines both of them
# Making it the final model
models = [model,model2]
```
Now models contains both part of the model, for output we will need to define our preprocessing function which will transform the output of the NeuralNet for our own needs.

For this example we use softmax for probability output of the 9 classes of the last layer.

```
# output step for neural net output
def prediction_preprocess(x):
    return torch.softmax(x,dim=1)
```


Now to assemble the model we need to define loss,optimizer the model
```
final_model = CustomModel(models,prediction_preprocess) # CustomModel(params).cuda() for gpu
optim = torch.optim.Adam(final_model.parameters(),lr=1e-4)
loss = nn.CrossEntropyLoss() #classification loss
```


In utils we have train and test_accuracy for training/testing our model, those functions can be replaced by any custom written functions.

