# GradientDescent
Implementation of Gradient Descent Optimization method with Python from scratch

Gradient descendent for linear regression
We already talk about linear regression which is a method used to find the relation between 2 variables. It will try to find a line that best fit all the points and with that line, we are going to be able to make predictions in a continuous set (regression predicts a value from a continuous set, for example, the price of a house depending on the size). How we are going to accomplish that? Well, we need to measure the error between each point to the line, find the slope and the alpha value to get the linear regression equation. 
 We talk about how we can find the best line using concepts as covariance, variance and the method of least squares.
Now we are going to find the line that best fit our data using Gradient Descent which is a method for optimization used in machine learning.
Gradient Descent
Gradient descent is a first order optimization method that means that it uses the first derivate to find local minuma, in more detail it uses partial derivate to find it. 
 
To start, let's suppose we have a simple quadratic function, f(x)=x2−6x+5, and we want to find the minimum of this function. 
We can derivate the function f’(x) = 2x-6 and we’ll find that the minimum is located in 3. But has we have already talked about, finding the minimum with really complex ecuation (real life ecuations) is not as easy. So we use the gradient descent to find local minima.
# Minimizing a quadratic function
import numpy as np
import matplotlib.pyplot as plt

x= np.linspace(-10,10,1000) #we generated 1000 point between -10 and 10
y = x**2 - 6*2 + 5  #f(x)=x2−6x+5

fig, xa = plt.subplots()
xa.plot(x,y)
#fig.show()

def function_derivate(x):
    return 2*x -6
minima =15
alpha = 0.01 #razon de aprendizaje
presition = 0.0001
move=1 #in this case any number > presition

while abs(move)> presition:
    gradient = function_derivate(minimo)
    move = gradient * alpha
    minima = minimo-move
    #print(move)

print("The minima is located in {}" .format(round(minimo,2)))
 
As you can see it is able to find the local minima. You can play with the values of presition, alpha to find the local minima. You need to be carefuly because if you select the value of alpha to small it will take more time and more power processing to find the minima in the other hand if you take a value to big it might loose the minima and ending in an infinite loop.

Lets do it step by step to understand how to find the line that best fits our data works.
First you have your data, of two variales that has a relation between them. And you can see that relashion more clearly if you plot them.
 
You can see that there`s a relation between that 2 variables. For example that 2 variables could be size of a house and house price.
We want to be able to predict the house price depending on the size. This is called regression because the result can be any value.  So we want to predict the price of a house given their size. To be able to predict the house price we use the data that we have and we draw a line that fits bet on all the points and what it means is that that line is going to be the one that the error is the slowest.
Ok, but now what does it mean that the error is the slowest? Well it means that the distance between the line and each point is the slowest. 
 
So, what we already have is a bunch of points that has a relation between them, we want to be able to predict base on the independent variable so we draw a line that best fit all the data points. But, how we do that?
We can define a line with the next equation
 
Now we know that that line that best fits our data needs to have the unique slope (m) and y-intercept (b). To be able to find those to variables we need to define a function to represent the error that are formed for those two variables. 
Error function SSE
   
Now we have a function that allow us to messure the error given a certain line. What we are gonna do is to choose randomly the value for m and for b. We are going to draw that line to see how that line passes through our points and also we are going to calculate the Error function. Lets do that with a python example. I`m gonna use a kaggle data set to find the relation between the raiting of a player base on his ADR.
Dependencies
 
dataset = pd.read_csv("HLTVData/playerStats.csv")
dataset.head(20)
 
As you can see  the csv file has some columns that we don`t need. So we fix the data.
dataSet = pd.read_csv("HLTVData/playerStats.csv",usecols=['ADR','Rating'])
dataSet['Rating'] = dataSet['Rating']*100
dataSet.head()
 
Now we plot our  data set.
X= np.array(dataSet['ADR'])
y = np.array(dataSet['Rating'])
plt.scatter(X,y)
plt.xlabel('ADR')
plt.ylabel('Rating')
plt.show()
 

Now we select a random value for m and for b and we plot the line.
m = 5
b = 3
plt.scatter(X,y)
plt.plot(X,m*X+b,color='red')
plt.show()
 
As we can see this line has a big error. And to be able to compute the Error we need to create a function that give us the SSE .
 
def SSE(m,b,data):
    totalError=0.0
    totalNan = 0
    for i in range(data.shape[0]):
        if(math.isnan(data[i,0])):
            totalNan +=1
        else:
            yOutput = m*data[i,0]+b
            y = data[i,1]
            error = (y-yOutput)**2
            totalError =totalError+ error
    return totalError
Now we can see the error value given for the SSE ecuation.
m = 5
b = 3

sse = SSE(m,b,data)
print('For the fitting line: y = %sx + %s\nSSE: %.2f' %(m,b,sse))
 

We now need to update the m and b values to minimize the error function. In order to do that we are going to need to run gradient descent on this error function. Since our error function is defined by the m and b parameters we are going to need to calculate the partial derivative for each parameter.
 
Before we have showed the Error function as the square error of the differences of the desire output and the actual input. If you had wondered why we square the error, well, this is why. Because we were going to need to compute the partial derivate and that helps us to make it easier.
Continuing with gradient descent our next step is to follow the equation. But what does the equation says?
I says that for each point we are going to compute the derivate, then we are going to sum that result and after that we are going to multiply that result by the negative stepper of the gradient. It means that we need to go in againts of the gradient. Remember that the stepper size is important, to small and it will take a lot of processing power and it will take a lot of time to find the local minima, to big I will loose the local minima. In each iteration  the function will update the m and b value to minimize the error. The function will be updating those parameters until a specific threshold is passed or until a specific certain of iteration are passed.
def gradient_descent_step(m,b,data):
    
    n_points = data.shape[0] #size of data
    m_grad = 0
    b_grad = 0
    stepper = 0.0001 #this is the learning rate
    
    for i in range(n_points):

        #Get current pair (x,y)
        x = data[i,0]
        y = data[i,1]
        if(math.isnan(x)|math.isnan(y)): #it will prevent for crashing when some data is missing
            #print("is nan")
            continue
        
        #you will calculate the partical derivative for each value in data
        #Partial derivative respect 'm'
        dm = -((2/n_points) * x * (y - (m*x + b)))
        
        #Partial derivative respect 'b'
        db = - ((2/n_points) * (y - (m*x + b)))
    
        #Update gradient
        m_grad = m_grad + dm
        b_grad = b_grad + db
    
    #Set the new 'better' updated 'm' and 'b'
    m_updated = m - stepper*m_grad
    b_updated = b - stepper*b_grad    
    return m_updated,b_updated

