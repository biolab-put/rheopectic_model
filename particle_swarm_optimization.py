
from numpy import arange


def banana(x):
    #Rosenbrock's banana function
    #https://en.wikipedia.org/wiki/Rosenbrock_function
    x1 = x[0]
    x2 = x[1]
    return (x1**4 - 2*x2*x1**2 + x2**2 + x1**2 - 2*x1 + 5)

def banana_with_arguments(x, A, B): # banana_with_arguments(x, *args): and then unpack args ( A, B = args )
    #Rosenbrock's banana function with arguments
    #https://en.wikipedia.org/wiki/Rosenbrock_function
    x1 = x[0]
    x2 = x[1]
    return (x1**4 - A*x2*x1**2 + x2**2 + x1**2 - A*x1 + B)

def bananas(x):
    #special form required by pyswarms
    x1 = x[:,0]
    x2 = x[:,1]
    return x1**4 - 2*x2*x1**2 + x2**2 + x1**2 - 2*x1 + 5

def con(x, *_): # *_ to express no interest in any further arguments
    x1 = x[0]
    x2 = x[1]
    return -(x1 + 0.25)**2 + 0.75*x2

def con_scikit(x):
    return -con(x)

lb = [-3, -1]
ub = [2, 6]


def scikit_opt_test():
    print(scikit_opt_test.__name__)
    from sko.PSO import PSO
    import matplotlib.pyplot as plt

    pso = PSO(func=banana, n_dim=2, pop=100, max_iter=100, lb=lb, ub=ub, w=0.5, c1=0.5, c2=0.5,constraint_ueq=(con_scikit,))
    pso.run()

    print('best_x is ', pso.gbest_x, 'best_y is', pso.gbest_y)
    #plt.plot(pso.gbest_y_hist)
    #plt.show()
    #best_x is  [0.79487092 0.45598025] best_y is [4.07299748]

def pyswarm_test():
    print(pyswarm_test.__name__)
    from pyswarm import pso
    #xopt, fopt = pso(banana, lb, ub, f_ieqcons=con,maxiter= 100)
    xopt, fopt = pso(banana_with_arguments, lb, ub, ieqcons =[con],maxiter= 100, args = (2,5))
    print('best_x is ', xopt, 'best_y is', fopt)


#Pyswarms rejected for no space for constrains
def pyswarms_test():
    print(pyswarms_test.__name__)
    import pyswarms as ps
    options = {'c1': 0.5, 'c2': 0.3, 'w':0.9}

    # Call instance of PSO
    optimizer = ps.single.GlobalBestPSO(n_particles=10, dimensions=2, options=options)
    cost, pos = optimizer.optimize(bananas, iters=1000)


#scikit_opt_test()
pyswarm_test()
#pyswarms_test()
