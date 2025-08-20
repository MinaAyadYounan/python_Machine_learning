# python_Machine_learning
This project is a simple implementation of Logistic Regression from scratch in Python.
It is designed for binary classification tasks.
Test the model on a real dataset: Breast Cancer Dataset from scikit-learn.
z= θ_0 + θ1*​x1​ + θ_2*​x_2 ​+ ⋯ + θ_n*​x_n​
Where:
θ₀ → Bias term
θ₁…θₙ → Parameters/weights for each feature
x₁…xₙ → Input features

Custom implementation of:
  Sigmoid function : 
    Used for binary classification (2 classes).
    Maps any real number to a probability in [0,1]
    mathematical formula is : σ(z) = 1/(1+ exp(−z))
optimization
We use Gradient Descent to update parameters:
  θ:=θ - α * ∇J(θ)
  Gradient calculation 
   get gradient descent by derivatives of cost function to theat we will get:
    ∇J(θ)=(1/m)*XT*(σ(Xθ)−y) 
    Where:
      m = number of training samples
      x = feature matrix (with bias column added)
      y = true labels
      α = learning rate
Prediction (probability & class label):
  using sigmoid function (σ(Xθ)) to get probability and by specific threshold defualt = 0.5  we predict if  tumor is malignant or not
Dataset:
  Breast Cancer Dataset from scikit-learn.
  Each row = a patient.
  Each column = a measured feature of the tumor (mean radius, mean texture, smoothness, etc.).
  Total features ≈ 30.
  Task: Predict whether the tumor is malignant (1) or benign (0).
After splitting the dataset:
  80% for training, 20% for testing.
  Data is standardized using StandardScaler.
Accuracy:
  Training Accuracy ≈ 98%
  Testing Accuracy ≈ 99%
