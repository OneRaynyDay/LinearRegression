import numpy as np

class LinearRegression:
    def __init__(self, matrix):
        '''X: The input of the supervised learning
        y: The results of the supervised learning'''
        biasTerm = np.ones((matrix.shape[0], 1))
        print("biasTerm's shape: " + str(biasTerm.shape))
        print(matrix)
        Xcomp = np.array(matrix[:,0:-1])
        print(Xcomp)
        self.X = np.append(biasTerm, Xcomp, axis=1)
        print("X's shape: " + str(self.X.shape))
        self.y = matrix[:,-1]
        print("Y's shape: " + str(self.y.shape))
        self.Theta = np.zeros((self.X.shape[1]))
        print("Theta's shape: " + str(self.Theta.shape))

    def costFunction(self):
        '''
        m: # of samples
        :return: COST of current theta
        '''
        m = self.X.shape[0]
        Cost = np.sum((np.dot(self.X,self.Theta) - self.y), axis=None)/(2*m)
        return Cost

    def gradDescent(self, alpha):
        m = self.X.shape[0]
        for i in range(m): #going through i-th sample
            print("Theta: " + str(self.Theta))
            print("shape: " + str(np.dot(self.X,self.Theta.T).shape))
            self.Theta = self.Theta - alpha*(np.sum((np.dot(self.X,self.Theta.T) - self.y), axis=None)*self.X[i])/m
            print("Theta: " + str(self.Theta))

