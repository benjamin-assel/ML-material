"""  Neural Network from scratch with numpy """


import numpy as np

np.random.seed(11)


class Neural_Network:

    def __init__(self, layers, activations, loss='mse'):

        assert len(layers)==len(activations)+1
        self.layers = layers
        self.activations = activations
        self.loss = loss
        self.weights = []
        self.biases = []

        for i in range(len(layers)-1):
            self.weights.append(np.random.uniform(-1,1,[layers[i], layers[i+1]]))
            self.biases.append(np.random.uniform(-1,1,[layers[i+1],1]))


    def feed_forward(self, X):
        """Feed forward prediction for a single input row x."""
        assert X.shape[1] == self.layers[0]
        Z = []
        A = [X.T]
        for i in range(len(self.activations)):
            Z.append((self.weights[i].T).dot(A[-1])+ self.biases[i])
            A.append(self.activ(self.activations[i])(Z[-1]))
        return Z, A

    @staticmethod
    def activ(func):
        """Returns the activation function."""
        if func == 'id':
            return lambda x: x
        elif func == 'relu':
            def relu(x):
                y = x
                y[y<0]=0
                return y
            return relu
        elif func == 'sigmoid':
            return lambda x: 1/(1+ np.exp(-x))
        else:
            print("Unknown activation function. Identity is used instead.")
            return lambda x: x


    def backprop(self, y, ZA):
        """Compute the gradients of W and b for one step of GD, assuming loss=mae."""
        assert self.loss == 'mse'
        Z, A = ZA
        gZ = []
        gW = []
        gb = []
        gA = [(A[-1]-y.T)/y.shape[0]]  # Derivative of the loss wrt A[-1]. Here we assume loss = mse

        for i in reversed(range(len(layers)-1)):
            gZi = self.derivative(activations[i])(Z[i])*gA[-1]
            #gZi = np.zeros(list(Z[i].shape))
            #for k in range(len(Z[i])):
            #    gZi[k][0] = self.derivative(activations[i])(Z[i][k][0])*gA[-1][k][0]
            gWi = A[i].dot(gZi.T)
            gbi = gZi.dot(np.ones([gZi.shape[1],1]))
            gAi = (self.weights[i]).dot(gZi)
            gZ.append(gZi)
            gW.append(gWi)
            gb.append(gbi)
            gA.append(gAi)

        return list(reversed(gW)), list(reversed(gb))

    @staticmethod
    def derivative(func):
        """Returns the derivative of the activation function."""
        if func == 'id':
            return lambda x: 1
        elif func == 'relu':
            def step(x):
                y = x
                y[y<0]=0
                y[y==0]=0.5
                y[y>0]=1
                return y
            return step
        elif func == 'sigmoid':
            return lambda x: 1/(2+np.exp(x)+np.exp(-x))
        else:
            print("Unknown activation function. Identity is used instead.")
            return lambda x: 1


    def train(self, X, y, batch_size, epochs, lr): # add X_val, y_val, num_epoch_max
        """."""
        num_batch = len(X)//batch_size
        best_loss = np.inf
        #count_epoch = 0 # for early stopping

        for n in range(epochs):
            for m in range(num_batch):
                X_batch = X[m*batch_size:(m+1)*batch_size]
                y_batch = y[m*batch_size:(m+1)*batch_size]
                ZA_batch = self.feed_forward(X_batch)
                #batch_loss = sum((ZA_batch[1][-1].T - y_batch)**2)/batch_size
                gW, gb = self.backprop(y_batch,ZA_batch)
                for i in range(len(self.layers)-1):
                    self.weights[i] +=  -lr*gW[i]
                    self.biases[i] += -lr*gb[i]

            _, A = self.feed_forward(X)
            loss = sum((A[-1].T - y)**2)[0]/len(y)
            if loss < best_loss:
                best_loss = loss

        print("Best loss: {}.".format(best_loss))

        #return best_loss


    def predict(self, X):
        _, A = self.feed_forward(X)
        return A[-1].T



# Test case

layers = [16,8,1]
activations = ['relu','sigmoid']
NN_0 = Neural_Network(layers, activations)

#print(NN_0.weights, end='\n')
#print(NN_0.biases, end='\n')
X = np.random.uniform(-1,1,[1000,16])
y = np.random.uniform(0,1,[1000,1])

#Z_0, A_0 = NN_0.feed_forward(X)
#print([A_0[i].shape for i in range(len(A_0))])
#dW_0 , db_0 = NN_0.backprop(y, NN_0.feed_forward(X))
#print(dW_0, db_0)
#print([dW_0[i].shape for i in range(len(dW_0))])

NN_0.train(X, y, batch_size=50, epochs=10, lr=0.2)
print(NN_0.weights, NN_0.biases)
