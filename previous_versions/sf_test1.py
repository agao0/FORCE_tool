import symforce

symforce.set_epsilon_to_symbol()

import symforce.symbolic as sf
from symforce.notebook_util import display, display_code, display_code_file, print_expression_tree
import json
from functools import reduce
from symforce.codegen import Codegen, CppConfig
from symforce.values import Values

# import numpy as np
# from sympy import *

############################################### PREMISE ####################################################
# Creating a homogenous transformation matrix using the product of exponentials formula with joint angles as
# symbolic variables, and generating C++ code of the resulting symbolic expression.

# GOAL: create a program that takes in a robot parameter file that specifies the general properties of the
# robot, and outputs a general function that represents a rigid body transformation (RBT) of the robot
# which uses symbolic expressions. This RBT will take in a list of angles: theta_n in order to generate 
# a configuration of the robot after rotation by this list of angles (corresponding to a kinematic
# description of the robot post movement).


################################################# SETUP ####################################################
# ASSUMING ONLY REVOLUTE JOINTS:
# We assume that we are given a robot parameter file, which contains:
# 
# a list of rotation axes: omega_n NOTE: NEED AXES TO HAVE MAGNITUDE 1
# a list of points on a corresponding rotation axes: q_n
# the initial configuration of the robot: g_st(0)
#
# ^the initial values of these will all be given as numbers.

# We suppose our fixed inertial frame is frame S and the final frame of the end effector frame is frame T


################################################ FUNCTIONS #################################################

def rotationWedge(vec):
# assume 3 x 1 col vector
    ret = sf.Matrix33()
    ret[0, 1] = -vec[2]
    ret[1, 0] = vec[2]
    ret[0, 2] = vec[1]
    ret[2, 0] = -vec[1]
    ret[2, 1] = vec[0]
    ret[1, 2] = -vec[0]
    return ret

# test = sf.Matrix([1, 2, 3])
# print(rotationWedge(test)) # checked

def generalWedge(vec):
# assume 6 x 1 col vector
    v = vec[:3]
    omega = vec[3: 6]
    omegaHat = rotationWedge(omega)
    ret = omegaHat.row_join(v)
    ret = ret.col_join(sf.Matrix14())
    return ret

def rodriguefy(omega, theta):
# returns e^(omega^hat * theta)
# assume omega is a symforce 3x1 vector, and theta is a symforce symbol
    ret = sf.Matrix.eye(3)
    wedge = rotationWedge(omega)
    temp1 = wedge*sf.sin(theta)
    temp2 = wedge*wedge*(1-sf.cos(theta))
    return ret + temp1 + temp2

# print(rodriguefy(sf.Matrix([1,2,3]), sf.Symbol('theta'))) # checked

def generalExponential(twist_coordinates, theta):
    v = twist_coordinates[:3]
    omega = twist_coordinates[3: 6]

    upperLeft = rodriguefy(omega, theta)

    temp = sf.Matrix.eye(3) - upperLeft
    temp = temp*(omega.cross(v))
    upperRight = temp+((omega*sf.Matrix.transpose(omega))*v*theta)
    ret = upperLeft.row_join(upperRight)
    ret = ret.col_join(sf.Matrix(1, 4, [0, 0, 0, 1]))
    return ret


################################################# MAIN ######################################################
# Parsing, JSON format assumed. The data.json file should have a 'joints' key, whose value is a list in the
# form: [[omega_1, q_1], [omega_2, q_2], [omega_3, q_3], ...] (each omega/q is also a list)

try:
    with open('test.json') as f:
        data = json.load(f)
        joints = data['joints']
        axes = [i[0] for i in joints]
        axes_points = [i[1] for i in joints]
        # axes = list(data['axes'].values())
        # axes_points = list(data['axes_points'].values())
        initial_configuration = sf.Matrix(data['init_config'])
# NOTE: Python 3.7 and greater, dictionaries are ordered, we will use this fact
except: 
    raise Exception("CHECK DATA FORMATTING") 

NUM_ANGLES = len(axes)

# Wrapper class to accomodate unknown number of joints, can only define here once NUM_ANGLES is determined
class myMatrix(sf.Matrix):
    SHAPE = (NUM_ANGLES, 1)

# Convert axes, axes_points to symforce matrices
axes = [sf.Matrix(i) for i in axes]
axes_points = [sf.Matrix(i) for i in axes_points]
# print(axes)
# print(axes_points)

# Create a matrix of joint angle symbols, [theta_n], that will be the primary input for the C++ function
joint_angles = sf.Matrix(list(sf.symbols(f"theta:{NUM_ANGLES}")))

# Generate twists coordinates
twists_coordinates = []
for i in range(NUM_ANGLES):
    temp = -axes[i]
    temp = temp.cross(axes_points[i])
    twists_coordinates += [temp.col_join(axes[i])]
# display(twists_coordinates)

# Function to generate product of exponentials that we will convert to C++ function
def myFunction(joint_angles: myMatrix) -> sf.Matrix44:
    expr = [generalExponential(twists_coordinates[i], joint_angles[i]) for i in range(NUM_ANGLES)]
    expr = reduce(lambda x, y: x*y, expr)
    expr = expr*initial_configuration
    return expr
    # print(expr)
    # print(generalWedge(twists_coordinates[0]))
    # print(generalExponential(twists_coordinates[0], sf.Symbol('theta')))

# Converting myFunction into a function in a C++ file
codegen = Codegen.function(func=myFunction, config=CppConfig())
codegen_data = codegen.generate_function(output_dir="/tmp/sf_codegen_myFunction_TESTING")
# On my mac, this was located in a hidden folder inside my main storage drive labeled "private" then "tmp/sf....."
print("Files generated in {}:\n".format(codegen_data.output_dir))
for f in codegen_data.generated_files:
    print("  |- {}".format(f.relative_to(codegen_data.output_dir)))

# display_code_file(codegen_data.generated_files[0], "C++")
# ^creates an IPython.core.display.HTML object, probably for use in Jupyter notebooks at the like


################################################# TESTING ######################################################
# Getting familiar with Symforce/Sympy

# pprint(Symbol('xi^'))

# theta = sf.Symbol('theta')
# print(sf.sin(theta))
##########

# x, y, z = sf.symbols('x y z')

# M = sf.Matrix(3, 1, [x, y, z])
# M2 = sf.Matrix(3, 1, [1, 2, 3])

# M3 = sf.Matrix.eye(3)
# M4 = sf.Matrix.eye(3)

# M5 = sf.Matrix([1,2,3,4,5])

# print(M*sf.Matrix.transpose(M))
# print(type(M5[3:10]))
# print(M3.row_join(M))
# print(M3*M4)
# print(M.cross(M2))

# print(-M) # nice

# M3 = sf.Matrix.eye(3)

# print(M3)
# print(M3[2,1])
# M3[2,1] = 5
# print(M3[2,1])
# print(M3)

# # print(M3.exp()) # doesn't work

# test = M3.row_join(M)
# display(test)
# print(test.shape)

# print(sf.exp(x)) # doesn't work with matrices, similarly, can't use matrix symbols in symforce

##########

# x = sf.Symbol("x")
# y = sf.Symbol("y")
# expr = x ** 2 + sf.sin(y) / x ** 2
# display(expr)
# display(expr.subs({x: 1.2, y: 0.4}))

# hatx = sf.Symbol("\hat{x}")
# display(hatx)

###########

# x = sf.Symbol('x')
# print(type(x.name))

###########

