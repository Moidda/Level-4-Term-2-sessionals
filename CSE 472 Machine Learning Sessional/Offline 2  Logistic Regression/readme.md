# Notes

## [Reading from csv file into Pandas Dataframe](https://pandas.pydata.org/docs/getting_started/intro_tutorials/02_read_write.html)

## [Splitting DataFrame by columns](https://www.activestate.com/resources/quick-reads/how-to-access-a-column-in-a-dataframe-using-pandas/#:~:text=You%20can%20use%20the%20loc,Let's%20see%20how.&text=If%20we%20wanted%20to%20access,in%20order%20to%20retrieve%20it.)

- Using iloc for integer indexing
- ```data.iloc[rowStart:rowEnd, colStart:colEnd]```
```python
import pandas as pd

data = pd.read_csv("data.csv")
features = data.iloc[:, :-1]
classes = data.iloc[:, -1:]
```

## [Splitting into train and test using pandas](https://pub.towardsai.net/3-different-approaches-for-train-test-splitting-of-a-pandas-dataframe-d5e544a5316)

## Study Links

- [realpython.com](https://realpython.com/logistic-regression-python/)
- [StatQuest Youtube](https://www.youtube.com/watch?v=yIYKR4sgzI8&ab_channel=StatQuestwithJoshStarmer)
- [Lecture by Andrew Ng](https://www.youtube.com/watch?v=-la3q9d7AKQ&list=PLNeKWBMsAzboR8vvhnlanxCNr2V7ITuxy&index=1&ab_channel=ArtificialIntelligence-AllinOne)
- [Implementation by Tutorials point](https://tutorialspoint.dev/language/python/understanding-logistic-regression)


## Terms

### Coefficients: $\theta$
The vector of coefficients for each feature

$$
\theta = 
    \begin{matrix}
    \theta_0 \\
    \theta_1 \\
    \vdots \\
    \theta_n
    \end{matrix}
$$

### Features: x
The vector of values of each features. For $n$ features, the matrix would be
$$
x = 
    \begin{matrix}
    x_0 \\
    x_1 \\
    \vdots \\
    x_n
    \end{matrix}
$$

For $m$ sets of data, the feature matrix is represented as
$$
X = 
    \begin{matrix}
    x_{0,0} & x_{0,1} & \cdots & x_{0, n} \\
    x_{1,0} & x_{1,1} & \cdots & x_{1, n} \\
            & \vdots  &        &         \\
    x_{m,0} & x_{m,1} & \cdots &x_{m, n} 
    \end{matrix}
$$


### Hypothesis, $h_\theta(x)$
The hypothesis is represented as,
$$
P(y_i | x_i; \theta) = h_\theta(x)^i = sigmoid(\theta^Tx)
$$
which denotes the probability that the class is $y_i$ given the features $x_i$ with the coefficients $\theta$



### Cost Function, $J(\theta)$
The cost function chosen is 
$$
Cost(h_{\theta}(x), y) = 
            \begin{matrix}
            -log(h_\theta(x)) \hspace{0.5cm}  & if \hspace{0.2cm} y=1 \\
            -log(1 - h_\theta(x)) \hspace{0.5cm} & if \hspace{0.2cm} y=0
            \end{matrix}
$$
which can be written in a single line as
$$
Cost(h_{\theta}(x), y) = - \{ ylog(h_\theta(x)) + (1-y)log(1 - h_\theta(x)) \}
$$

We write $J(\theta)$ as the mean $Cost$ over all the $m$ data set

$$
J(\theta) = 1/m * \sum_{i=1}^{m} Cost(h_\theta(x)^{i}, y^i) 
$$


### Gradient Descent
Simultaneously update for all values of $j$
$$
\theta_j = \theta_j - \alpha \frac{\partial }{\partial \theta_j} J(\theta)
$$

$$
\frac{\partial }{\partial \theta_j} J(\theta) = \sum_{i=1}^{m}(h_\theta(x)^i-y^i)x_j^i
$$

Here, $\alpha$ is the learning rate


## Training Steps
- Initialize the vector of coefficients, the $\theta$ vector. Fill it with 0s

- Run gradient descent to find out the $\theta$ vector that minimizes $J(\theta)$

- Use the $\theta$ vector found from gradient descent to predict test data


