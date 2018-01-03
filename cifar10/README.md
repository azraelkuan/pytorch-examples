# A Home Work For ESL



## Require

+ Python3.2+
+ pytorch
+ scikit-learn
+ tqdm



## BaseLine

### KNN

run knn.py, this may cost u 20 cpus. `python knn.py | tee logs/knn.log`

>  knn parameters: 

+ Neightboors : `1, 3, 5, 7, 9, 10, 13, 17, 20, 50, 75, 100`
+ Weights: `uniform, distance`

> Final Result:

in the logs/knn.log



## SVM

run svm.py, also needs 20 cpus. `python svm.py | tee logs/svm.log`

I use sgd to train the svm, so serveral parameters as below:

> SVM parameters:

+ alpha: Constant that multiplies the regularization term
+ learning rate: options for lr
+ eta0: the value of learning rate
+ max_iter: the maximum number of passes over the training data 
+ average: the batch size



## NNET

use DNN and ResNet. 

> Result: logs/*.log

U can also define your own model in model.py

**Need One GPU**: `python train.py | tee logs/xx.log`