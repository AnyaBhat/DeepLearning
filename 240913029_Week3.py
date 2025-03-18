"""import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm,bernoulli
import matplotlib.pyplot as plt

class MLEMAPExamples:
    def __init__(self):
        self.generate_classification_data()
        self.generate_regression_data()
    def generate_classification_data(self):
        true_p=0.7
        self.n_samples_cls=100
        self.X_cls=bernoulli.rvs(true_p,size=self.n_samples_cls)

    def generate_regression_data(self):
        true_params={'beta':2.5,'sigma':1.0}
        self.n_samples_reg=50
        self.X_reg=np.linspace(0,10,self.n_samples_reg)
        noise=np.random.normal(0,true_params['sigma'],self.n_samples_reg)
        self.y_reg=true_params['beta']*self.X_reg+noise

    def classification_mle(self):
        p_mle=np.mean(self.X_cls)
        return p_mle

    def classification_map(self,prior_alpha=10,prior_beta=10):
        successes=np.sum(self.X_cls)
        n=len(self.X_cls)
        p_map=(prior_alpha+successes-1)/(prior_alpha+prior_beta+n-2)
        return p_map
    def regression_neg_likelihood(self,params,X,y):
        beta,sigma=params
        mu=beta*X
        return -np.sum(norm.logdf(y,mu,sigma))

    def regression_neg_log_posterior(self,params,X,y,prior_mu,prior_sigma):
        beta,sigma=params
        nll=self.regression_neg_likelihood(params,X,y)
        log_prior_beta= -0.5*((beta-prior_mu['beta'])/prior_sigma['beta'])**2
        log_prior_sigma=-2*np.log(sigma)
        return nll -log_prior_beta-log_prior_sigma

    def regression_mle(self):
        initial_params=[1.0,1.0]
        result=minimize(
            self.regression_neg_likelihood,
            initial_params,
            args=(self.X_reg,self.y_reg),
            method='Neldor-Mead'
        )
        return result.x
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import seaborn as sns
from numpy.ma.core import indices
from sympy.physics.vector import gradient


class GradientDescentComparison:
    def __init__(self,n_samples=1000):
        np.random.seed(42)
        self.n_samples=n_samples
        self.X=np.random.uniform(-5,5,n_samples)
        self.y=2*self.X**2+3*self.X+np.random.normal(0,1,n_samples)
        self.x_plot=np.linspace(-5,5,100)
        self.history={
            'batch':{'params':[],'loss':[]},
            'stochastic':{'params':[],'loss':[]},
            'mini_batch':{'params':[],'loss':[]}
        }

    def true_function(self,x):
        return 2*x**2+3*x
    def predict(self,X,params):
        a,b=params
        return a*X**2+b*X
    def compute_gradients(self,X,y,params):
        a,b=params
        y_pred=self.predict(X,params)
        error=y_pred -y
        grad_a=np.mean(2*error*X**2)
        grad_b=np.mean(2*error*X)

        return np.array([grad_a,grad_b])

    def compute_loss(self,X,y,params):
        y_pred=self.predict(X,params)
        return np.mean((y_pred-y)**2)

    def batch_gradient_descent(self,learning_rate=0.001,n_iterations=100):
        params=np.array([0.1,0.1])
        for i in range(n_iterations):
            gradients=self.compute_gradients(self.X,self.y,params)
            params=params-learning_rate*gradients
            self.history['batch']['params'].append(params.copy())
            self.history['batch']['loss'].append(
                self.compute_loss(self.X,self.y,params)
            )
        return params

    def stochastic_gradient_descent(self,learning_rate=0.001,n_iteration=100):
        params=np.array([0.1,0.1])

        for i in range(n_iteration):
            for j in range(self.n_samples):
                idx=np.random.randint(0,self.n_samples)
                X_samples=self.X[idx]
                y_samples=self.y[idx]
                gradients=self.compute_gradients(
                    np.array([X_samples]),
                    np.array([y_samples]),
                    params
                )
                params=params-learning_rate*gradients
            self.history['stochastic']['params'].append(params.copy())
            self.history['stochastic']['loss'].append(
                self.compute_loss(self.X,self.y,params)
            )
        return params

    def mini_batch_gradient_descent(self,batch_size=32,learning_rate=0.001,n_iteration=100):
        params=np.array([0.1,0.1])
        n_batches=self.n_samples//batch_size
        for i in range(n_iteration):
            indices=np.random.permutation(self.n_samples)

            for j in range (n_batches):
                batch_indices=indices[j*batch_size:(j+1)*batch_size]
                X_batch = self.X[batch_indices]
                y_batch = self.y[batch_indices]

                gradients=self.compute_gradients(X_batch,y_batch,params)
                params=params-learning_rate*gradients
            self.history['mini_batch']['params'].append(params.copy())
            self.history['mini_batch']['loss'].append(
                self.compute_loss(self.X, self.y, params)
            )
        return params

    def plot_results(self):
        plt.figure(figsize=(15,5))
        plt.subplot(1,2,1)
        plt.scatter(self.X,self.y,alpha=0.3,label='Data points')

        plt.plot(self.x_plot,self.true_function(self.x_plot),'k',label='true function')

        for method in ['batch','stochastic','mini_batch']:
            final_params=self.history[method]['params'][-1]
            plt.plot(self.x_plot,self.predict(self.x_plot,final_params),'--',label=f'{method.capitalize()} GD fit')

        plt.xlabel('X')
        plt.ylabel('y')
        plt.title('Data and Fitted Curves')
        plt.legend()

        plt.subplot(1,2,2)
        for method in ['batch', 'stochastic', 'mini_batch']:
            plt.plot(self.history[method]['loss'],
                     label=f'{method.capitalize()} GD')
        plt.xlabel('Iteration')
        plt.ylabel('MSE Log')
        plt.title('Learning Curves')
        plt.legend()
        plt.yscale('log')

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    gd=GradientDescentComparison()

    batch_params=gd.batch_gradient_descent()
    sgd_params=gd.stochastic_gradient_descent()
    mini_batch_params=gd.mini_batch_gradient_descent()

    print("\nFinal Parameters (a,b):")
    print(f"Batch GD :{batch_params}")
    print(f"SGD:{sgd_params}")
    print(f"Mini-batch-GD:{mini_batch_params}")

    gd.plot_results()
