This model is using MNIST dataset and trying to achieve 99% accuracy. 
I have 3 conv layers and 2 fc layers. 
I have a simple SGD model as optimizer and learning rate is adjusted based on the epoch .
Tried adding l1 and l2 regularization but it didnt help much with accuracy improvements. 
Tried sepearating the outliers using z_score_outlier calculation , but thats next TODO , didnt workout somehow. 
