# L-BFGS-Based-Adversarial-Input-Against-SVM-

# Data Source

# Method
## L-BFGS
L-BFGS method is developed from Newton's method. Compared with Newton's method, L-BFGS optimization achieves resource-efficient in both computation and storage.
Please check the report for more details.

# Parameters
There are three parameters are subject to change:
## x 
Is the target picture to be attacked

eg.



## y_prime
Is the target label 

eg.



## c
Is the perturbations degree of the attack

if c is too high, the perturbations may be too little to success

if c is too low, the perturbations may be too high to generate the attack target label

eg.

c is usually between 0.001 to 1, we have set the default c as 0.01

# Result 
For example, this picture shows an successful adversarial attack to mislead the system to predict 3 as 5

the left picture is original, 

the mid picture is the perturbation we add for adversarial attack, 

the right picture is the output
![alt text](https://user-images.githubusercontent.com/2645110/34134760-aa82300a-e42a-11e7-81a1-54e86d21b59e.png)

And here is an successful result for attacking image 5 to label 4,6,8
![alt text](https://user-images.githubusercontent.com/2645110/34134983-e7f44544-e42b-11e7-9e7d-c678701b91fa.png)

# Conclusion

