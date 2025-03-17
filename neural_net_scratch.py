import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets

X, y = sklearn.datasets.make_moons(500, noise=0.30)


class neural_network_complecated():
    def __init__(self,neuron_list , classes , lam):
        self.lamda = lam
        self.class_count = classes
        self.neuron_list = neuron_list
        self.layer1 = neuron_list[0]
        self.layer2 = neuron_list[1]
        self.layer3 = neuron_list[2]
        self.layer4 = neuron_list[3]
        super(neural_network_complecated, self).__init__()
        
    def sigmoid(self,p):
        y = 1/(1 + np.exp(-p))
        return y
        
    def relu(self,q):
        y = np.maximum(0,q)
        return y

    def tanh(self, p):
        return np.tanh(p)
        
    def sigmoid_derivative(self,p):
        r = self.sigmoid(p) * (1 - self.sigmoid(p))
        return r
        
    def relu_derivative(self,j):
        return np.where(j > 0, 1, 0)

    def tanh_derivative(self, p):
        return 1 - np.tanh(p) ** 2
            
    def softmax(self,x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    
    def lossfn(self, predicted, labels, weights):
        batch_size = predicted.shape[0]
        log_probs = -np.log(predicted[np.arange(batch_size), labels])
        loss = np.sum(log_probs) / batch_size
        regularization_loss = 0
        for w in weights:
            regularization_loss += np.sum(np.square(w))
        loss_net = loss + self.lamda * regularization_loss
        return loss_net
    
    def one_hot_encode(self, labels):
        one_hot_matrix = np.zeros((len(labels), self.class_count))
        one_hot_matrix[np.arange(len(labels)), labels] = 1
        return one_hot_matrix


    def weight_initialization(self, x):
        input_nodes = x.shape[1]
        output_nodes = self.class_count
        # Xavier/Glorot initialization for better convergence
        w1 = np.random.randn(input_nodes, self.layer1) * np.sqrt(2.0 / input_nodes)
        b1 = np.zeros((1, self.layer1))  
        w2 = np.random.randn(self.layer1, self.layer2) * np.sqrt(2.0 / self.layer1)
        b2 = np.zeros((1, self.layer2))
        w3 = np.random.randn(self.layer2, self.layer3) * np.sqrt(2.0 / self.layer2)
        b3 = np.zeros((1, self.layer3))
        w4 = np.random.randn(self.layer3 ,self.layer4 ) * np.sqrt(2.0 / self.layer3)
        b4 = np.zeros((1,self.layer4))
        w5 = np.random.randn(self.layer4 , output_nodes) * np.sqrt(2.0 / self.layer4)
        b5 = np.zeros((1,output_nodes))
        weights = [w1, w2, w3 ,w4 ,w5]

        
        biases = [b1, b2, b3,b4,b5]
        return weights, biases
        
    def forward(self,x,labels , weights , biases , activation):
        hidden_layers = len(self.neuron_list)
        z = x
        matrix_A = [x]
        matrix_z = []
        for layer in range(hidden_layers):
            z = np.dot(z , weights[layer]) + biases[layer]
            matrix_z.append(z)
            if activation == "sigmoid":
                a = self.sigmoid(z)
                
            if activation == "relu":
                a = self.relu(z)
                
            elif activation == "tanh":
                a = self.tanh(z)
                
            matrix_A.append(a)
        last_layer_op = np.dot(a,weights[-1])
        probs = self.softmax(last_layer_op)
        return probs ,matrix_A, matrix_z
        
    def backpropagation(self,probs,labels,matrix_z, matrix_A, weights, biases , lr , activation):
        batch_size = probs.shape[0]
        one_hot_labels = self.one_hot_encode(labels)
        delta = probs - one_hot_labels
        dldw3 = np.dot(matrix_A[-1].T , delta) / batch_size
        db3 = np.sum(delta, axis=0, keepdims=True) / batch_size
        #matrix_A.pop()
        weight_changer = [dldw3]
        bias_changer = [db3]
        #
        for i in range(len(matrix_z)-1 , -1, -1):
            if activation == "sigmoid":
                delta = np.dot(delta, weights[i+1].T) * (self.sigmoid_derivative(matrix_z[i]))
            if activation == "relu":
                delta = np.dot(delta, weights[i+1].T) * (self.relu_derivative(matrix_z[i]))
                
            elif activation == "tanh":
                delta = np.dot(delta, weights[i+1].T) * (self.tanh_derivative(matrix_z[i]))
            
            db = np.sum(delta, axis=0, keepdims=True) / batch_size
            bias_changer.append(db)
            dw = np.dot(matrix_A[i].T, delta)/ batch_size
            weight_changer.append(dw)
        
        bias_changer.reverse()
        weight_changer.reverse()
        #print(weight_update)
        for i in range(len(weights)):
            weights[i] = weights[i] - lr * weight_changer[i] - lr * self.lamda * weights[i]
            biases[i] = biases[i] - lr * bias_changer[i]
            
        return weights , biases

def plot_decision_boundary(X, y, model, weights, biases, activation):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    probs, _, _ = model.forward(grid_points, "pppp", weights, biases ,activation)
    predicted_labels = np.argmax(probs, axis=1)
    predicted_labels = predicted_labels.reshape(xx.shape)
    plt.contourf(xx, yy, predicted_labels, alpha=0.8, cmap=plt.cm.RdYlBu)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.RdYlBu)
    plt.title("Decision Boundary with Scattered Data Points")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()


model = neural_network_complecated([16,8,6,4], 2, 0)
weights, biases = model.weight_initialization(X)
weight_initial = [w.copy() for w in weights]
bias_initial = [b.copy() for b in biases]
loss_list_moon = []
num_epochs = 100000
for epoch in range(num_epochs):
    loss = 0
    probs, A, Z = model.forward(X,y, weights, biases ,  "tanh")
    weights, biases = model.backpropagation(probs, y, Z, A, weights, biases, 0.001, "tanh")
    loss += model.lossfn(probs, y, weights)
    loss_list_moon.append(loss)
    if epoch %10000 ==0 :
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss:.4f}")
        
    
weight_updated = [w.copy() for w in weights]
bias_updated = [b.copy() for b in biases]

plot_decision_boundary(X, y, model, weight_updated, bias_updated , "tanh")