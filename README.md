# Medical-Data-Neural-Net

Medical data from Erlanger is trained on by a basic neural net and then the neural net tries to classify a test set.

The training set consists of 1608 patients' data of various demographics and medical tests. The goal of the training is to predict wether or not the patient experienced acute coronary syndrome in the following 30 days. After training, it classifies the training set officially, and then classifies a new test set of 540 patients. 

The program also does analysis of the results giving the ROC points as output into cutoffs.txt which I have converted into .odc files and also two PNGs of the scatter plot. In terminal it outputs the area under the ROC which is a common metric for success of neural net classification. It also outputs the final weight values in no real order, just to see what the values were for general debugging. 
