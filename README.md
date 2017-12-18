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

## y_prime
Is the target label 

## c
Is the perturbations degree of the attack

if c is too high, the pertubations may be too little to success

if c is too low, the pertubations may be too high to generate the attack target label

c is usually between 0.001 to 1, we have set the default c as 0.01

# Result 


# Conclusion
