import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt

def dlqr(A,B,Q,R):
    """
    Solve the discrete time lqr controller.
    x[k+1] = A x[k] + B u[k]
    cost = sum x[k].T*Q*x[k] + u[k].T*R*u[k]
    """
    # first, solve the ricatti equation
    P = np.matrix(scipy.linalg.solve_discrete_are(A, B, Q, R))
    # compute the LQR gain
    K = np.matrix(scipy.linalg.inv(B.T*P*B+R)*(B.T*P*A))
    return -K
  
  
dt = 0.05

A = np.matrix([[1, 0],[0 , 1]])
B = np.matrix([[dt, 0], [0, dt]])

Q = np.matrix("1 0; 0 1")
R = np.matrix("0.01 0; 0 0.01")

K = dlqr(A,B,Q,R)


nsteps = 250
time = np.linspace(0, 2, nsteps, endpoint=True)
xk = np.matrix("5 ; 20")
xd = xd = np.matrix("1; 5")

X = []
Y = []
T = []
U = []
for t in time:
    xe = xk - xd
    uk = K*xe
    X.append(xk[0, 0])
    Y.append(xk[1, 0])

    xk = A*xk + B*uk

plt.plot(X, Y, label="car position, meters")
# plt.plot(time, T, label='pendulum angle, radians')
# plt.plot(time, U, label='control voltage, decavolts')

plt.legend(loc='upper right')
plt.grid()
plt.show()
