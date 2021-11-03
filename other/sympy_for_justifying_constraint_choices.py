from sympy import symbols, cos, sin, Matrix, Dummy, pi, asin, sqrt, simplify, Derivative, Abs, solve
from sympy.utilities.lambdify import lambdify
import numpy as np

def rad(deg):
    return deg*np.pi/180
def deg(rad):
    return rad*180/np.pi


U = symbols('U', real=True, positive=True)
u = symbols('u', real=True)
v = symbols('v', real=True)
w = symbols('w', real=True)
bb = symbols(r' bb', real=True, positive=True)  # CHANGE THIS to prove it works for both positive and negative
tb = symbols(r'tb', real=True)
costb = Dummy('costb', real=True, positive=True)

def R_x(alpha):
    return Matrix([[1, 0, 0],
                   [0, cos(alpha), -sin(alpha)],
                   [0, sin(alpha), cos(alpha)]])

def R_y(alpha):
    return Matrix([[cos(alpha), 0, sin(alpha)],
                   [0, 1, 0],
                   [-sin(alpha), 0, cos(alpha)]])

def R_z(alpha):
    return Matrix([[cos(alpha), -sin(alpha), 0],
                   [sin(alpha), cos(alpha), 0],
                   [0, 0, 1]])

T_LsGwb = ( R_y(tb) @ R_z(-bb-pi/2) ).T

# Mean system
U_Gw = Matrix([U, 0, 0])
U_Ls = T_LsGwb @ U_Gw
U_x = U_Ls[0]
U_y = U_Ls[1]
U_z = U_Ls[2]
U_yz = sqrt(U_y**2 + U_z**2)

theta_yz = simplify(asin(U_z/U_yz))
# theta_yz_2 = asin(sin(tb)/sqrt(1 - sin(bb)**2*cos(tb)**2))
# simplify(theta_yz - theta_yz_2)

theta_yz_symb = symbols("theta_yz_symb", real=True)

# solve(theta_yz_symb - cos(tb)**2, tb)
# solve(theta_yz_symb - asin(sin(tb)/sqrt(1 - sin(bb)**2*cos(tb)**2)), tb)

theta_yz_db = simplify(simplify(simplify(Derivative(theta_yz,bb).doit()).subs(cos(tb),costb)).subs(costb,cos(tb)))


simplify((U_yz**2 / U**2))
# Proof that theta_yz fulfils the derivative constraints of the polynomials
theta_yz_db.subs(bb,0)
theta_yz_db.subs(bb,pi/2)
theta_yz_db.subs(tb,pi/6).subs(bb,pi/2-0.000001)
theta_yz_db.subs(tb,0).subs(bb,pi/2)
theta_yz_db.subs(tb,0.001).subs(bb,pi/2)

Derivative(simplify(U_yz**2 / U**2), bb).doit().subs(bb,0)
Derivative(simplify(U_yz**2 / U**2), bb).doit().subs(bb,pi/2)
Derivative(simplify(U_x**2 / U**2), bb).doit().subs(bb,0)
Derivative(simplify(U_x**2 / U**2), bb).doit().subs(bb,pi/2)
simplify((U_yz**2 / U**2).subs(bb,pi/2))


Derivative(simplify(U_yz**2 / U**2  +  5*U_x**2 / U**2), bb).doit().subs(bb,pi/2)

Derivative(simplify(U_yz**2 +  5*U_x**2), bb).doit().subs(bb,pi/2)

def theta_yz_fun(beta, theta, U=30):
    return np.arcsin(U*np.sin(theta)/np.sqrt(U**2*np.sin(theta)**2 + U**2*np.cos(beta)**2*np.cos(theta)**2))

def theta_yz_db_fun(beta, theta, U=30):
    return (np.cos(2*beta - theta) - np.cos(2*beta + theta))*np.cos(theta)/(4*(np.sin(beta)**2*np.sin(theta)**2 - np.sin(beta)**2 + 1)*np.abs(np.cos(beta)))

# betas = np.linspace(-np.pi/6, np.pi/2+np.pi/6, 100)
# thetas = np.linspace(-np.pi/2, np.pi/2, 100)

# betas, thetas = np.meshgrid(np.linspace(-np.pi/6, np.pi/2+np.pi/6, 1000), np.linspace(-np.pi/2, np.pi/2, 1000))

betas, thetas = np.meshgrid(np.linspace(rad(80), rad(100), 1000), np.linspace(rad(-10), rad(10), 1000))

grid = np.meshgrid(deg(np.linspace(rad(80), rad(100), 1000)), deg(np.linspace(rad(-10), rad(10), 1000)))
thetas_yz = theta_yz_fun(betas,thetas)
thetas_yz_deg = deg(thetas_yz)

thetas_yz_db = theta_yz_db_fun(betas,thetas)
thetas_yz_db = thetas_yz_db


import matplotlib.pyplot as plt
plt.subplot(1,2,1)
plt.contourf(deg(betas), deg(thetas), deg(theta_yz_fun(betas,thetas)), 256, cmap='Spectral')
plt.colorbar()
plt.subplot(1,2,2)
plt.contourf(deg(betas), deg(thetas), thetas_yz_db, 256, cmap='jet', vmin=-10000, vmax=10000)
plt.colorbar()

theta_yz_db.subs(bb,pi/2)
theta_yz_db_fun(betas, thetas)

test = deg(theta_yz_fun(betas,thetas))


# Showing theta_yz_fun is not differentiable at beta=90 & theta!=0:
thetas = rad(-10) # deg
betas = rad(np.arange(0,180,1))

plt.plot(deg(betas), deg(theta_yz_fun(betas, np.ones(betas.shape)*thetas)))


