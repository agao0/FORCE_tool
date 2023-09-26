import symforce

symforce.set_epsilon_to_symbol()

import symforce.symbolic as sf
from symforce.notebook_util import display, display_code, display_code_file, print_expression_tree
import json
from functools import reduce
from symforce.codegen import Codegen, CppConfig
from symforce.values import Values

############################################### PREMISE ####################################################
# We are working with a robot manipulator and producing common expressions involved in manipulators. The
# key point of this program is to use symbolic variables in the expressions so that we can generate C++
# code of the resulting symbolic expression using symforce, which will be useful for RTC (real-time control).

# SPECIFIC EXPRESSIONS TO CREATE:
# forward kinematics map
# Adjoint map 
# Adjoint inverse map
# Jacobian map
# Velocity (?)
# Screws (?)
# MORE FRAMES = POINTS OF INTEREST

################################################# SETUP ####################################################
# Data formatted so that prismatic and revolute joints allowed.

# We assume that we are given a robot parameter file, which contains:
#
#   -> a list of twists corresponding to the twists associated with each joint IN REFERENCE CONFIG
#   -> the reference configuration of the manipulator: g_st(0)
#   -> the desired configuration of the manipulator: theta_n, an element of the joint space Q
#
# ^the initial values of these will all be given as numbers.

# We suppose our fixed inertial/base frame is frame S and the end effector/tool/body frame is frame T
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

def revoluteTwist(twist):
    pass

def CHECK_FORMATTING():
    pass

################################################# MAIN ######################################################
# Parsing, JSON format assumed. The data.json file should have a 'joints' key, whose value is a list in the
# form: [[omega_1, q_1], [omega_2, q_2], [omega_3, q_3], ...] (each omega/q is also a list)

try:
    with open('test2.json') as f:
        data = json.load(f)
        joint_twist_data = data['joint_twist_data']
        ref_config = sf.Matrix(data['ref_config'])
        generateAdjoint = data["generateAdjoint"]
        generateAdjointInverse = data["generateAdjointInverse"]
        generateJacobian = data["generateJacobian"]
        pointsOfInterest = data["pointsOfInterest"]
except: 
    raise Exception("CHECK DATA FORMATTING") 

NUM_ANGLES = len(joint_twist_data)

# Wrapper classes to accomodate unknown number of joints, can only define here once NUM_ANGLES is determined
class anglesMatrix(sf.Matrix):
    SHAPE = (NUM_ANGLES, 1)

class JacobianMatrix(sf.Matrix):
    SHAPE = (6, NUM_ANGLES)

if pointsOfInterest:
    class POIMatrix(sf.Matrix):
        SHAPE = (pointsOfInterest["jointNum"], 1)

# Create a list of twist coordinates and twists
joint_twist_coordinates = []
joint_twists = []
for i in joint_twist_data:
    if i[0] == "r":
        omega = sf.Matrix(i[1])
        n_omega = -1*omega
        q = sf.Matrix(i[2])
        v = n_omega.cross(q)
        t_coords = v.col_join(omega)
    elif i[0] == "p":
        omega = sf.Matrix(i[1])
        t_coords = omega.col_join(sf.Matrix31([0, 0, 0]))
    joint_twist_coordinates += [t_coords]
    joint_twists += [generalWedge(t_coords)]

# print(joint_twist_coordinates)
# print(joint_twist_coordinates[0].shape)
# print(joint_twists)

# Create a matrix of joint angle symbols, [theta_n], that will be the primary input for the C++ function
# joint_angles = sf.Matrix(list(sf.symbols(f"theta:{NUM_ANGLES}")))
# I DON'T THINK THIS DOES ANYTHING

# Function to generate product of exponentials that we will convert to C++ function
def forwardKinematicsMap(joint_angles: anglesMatrix) -> sf.Matrix44:
    expr = [generalExponential(joint_twist_coordinates[i], joint_angles[i]) for i in range(NUM_ANGLES)]
    expr = reduce(lambda x, y: x*y, expr)
    expr = expr*ref_config
    return expr

# Function to generate adjoint map corresponding to above POE formula that we will convert to C++ function
def adjointMap(joint_angles: anglesMatrix) -> sf.Matrix66:
    expr = forwardKinematicsMap(joint_angles)
    R = expr[0:3, 0:3]
    p = expr[0:3, 3]
    temp = rotationWedge(p) * R
    temp = R.row_join(temp)
    temp2 = sf.Matrix.zeros(3,3).row_join(R)
    ret = temp.col_join(temp2)
    return ret

# Function to generate inverse adjoint map that we will convert to C++ function
def inverseAdjointMap(joint_angles: anglesMatrix) -> sf.Matrix66:
    expr = forwardKinematicsMap(joint_angles)
    R = expr[0:3, 0:3]
    Rt = R.transpose()
    p = expr[0:3, 3]
    temp = -Rt * rotationWedge(p)
    temp = Rt.row_join(temp)
    temp2 = sf.Matrix.zeros(3,3).row_join(Rt)
    ret = temp.col_join(temp2)
    return ret

def generalAdjointMap(g: sf.Matrix44) -> sf.Matrix66:
    R = g[0:3, 0:3]
    p = g[0:3, 3]
    temp = rotationWedge(p) * R
    temp = R.row_join(temp)
    temp2 = sf.Matrix.zeros(3,3).row_join(R)
    ret = temp.col_join(temp2)
    return ret

def generalInverseAdjointMap(g: sf.Matrix44) -> sf.Matrix66:
    R = g[0:3, 0:3]
    Rt = R.transpose()
    p = g[0:3, 3]
    temp = -Rt * rotationWedge(p)
    temp = Rt.row_join(temp)
    temp2 = sf.Matrix.zeros(3,3).row_join(Rt)
    ret = temp.col_join(temp2)
    return ret

# Is this computed every time? Where in the computation chain does symforce actually optimize? (depends on how the program is used)
# Ask about double pendulum example which actually uses symbols

# Function to generate J^S_ST(theta) that we will convert to C++ function
def JacobianMap(joint_angles: anglesMatrix) -> JacobianMatrix:
    ret = joint_twist_coordinates[0]
    current_g = sf.Matrix.eye(4, 4)

    for i in range(0, NUM_ANGLES-1):
        current_g = current_g * generalExponential(joint_twist_coordinates[i], joint_angles[i])
        next_ret = generalAdjointMap(current_g)*joint_twist_coordinates[i + 1]
        ret = ret.row_join(next_ret)

    return ret

# My approach may be inefficient, could consider differentiating and using chain rule instead


# Converting forwardKinematicsMap into a function in a C++ file
codegen = Codegen.function(func=forwardKinematicsMap, config=CppConfig())
codegen_data = codegen.generate_function(output_dir="/tmp/sf_codegen_forwardKinematicsMap")
print("Forward Kinematics Map generated in {}:".format(codegen_data.output_dir))
for f in codegen_data.generated_files:
    print("  |- {}\n".format(f.relative_to(codegen_data.output_dir)))

if generateAdjoint:
    codegen = Codegen.function(func=adjointMap, config=CppConfig())
    codegen_data = codegen.generate_function(output_dir="/tmp/sf_codegen_adjointMap")
    print("Adjoint Map generated in {}:".format(codegen_data.output_dir))
    for f in codegen_data.generated_files:
        print("  |- {}\n".format(f.relative_to(codegen_data.output_dir)))

if generateAdjointInverse:
    codegen = Codegen.function(func=inverseAdjointMap, config=CppConfig())
    codegen_data = codegen.generate_function(output_dir="/tmp/sf_codegen_inverseAdjointMap")
    print("Inverse Adjoint Map generated in {}:".format(codegen_data.output_dir))
    for f in codegen_data.generated_files:
        print("  |- {}\n".format(f.relative_to(codegen_data.output_dir)))

if generateJacobian:
    codegen = Codegen.function(func=JacobianMap, config=CppConfig())
    codegen_data = codegen.generate_function(output_dir="/tmp/sf_codegen_JacobianMap")
    print("Jacobian Map generated in {}:".format(codegen_data.output_dir))
    for f in codegen_data.generated_files:
        print("  |- {}\n".format(f.relative_to(codegen_data.output_dir)))


######################################### POINTS OF INTEREST ###################################################
if pointsOfInterest:
    def forwardKMPOI(joint_angles: POIMatrix) -> sf.Matrix44:
        expr = [generalExponential(joint_twist_coordinates[i], joint_angles[i]) for i in range(pointsOfInterest["jointNum"])]
        expr = reduce(lambda x, y: x*y, expr)
        expr = expr*sf.Matrix(pointsOfInterest["ref_config"])
        return expr
    
    codegen = Codegen.function(func=forwardKMPOI, config=CppConfig())
    codegen_data = codegen.generate_function(output_dir="/tmp/sf_codegen_forwardKMPOI")
    print("Forward Kinematics Map for the POI generated in {}:".format(codegen_data.output_dir))
    for f in codegen_data.generated_files:
        print("  |- {}\n".format(f.relative_to(codegen_data.output_dir)))

################################################# TESTING ######################################################
# # Matrix slicing

# r1 = sf.Matrix([1,2,3,4]).transpose()
# r2 = sf.Matrix([5, 6, 7, 8]).transpose()
# r3 = sf.Matrix([9, 10, 11, 12]).transpose()

# M = r1.col_join(r2).col_join(r3)
# print(M)

# slice = M[0:3, 0:3]
# print(slice)

# # slice2 = M[0:3, 3]
# # print(slice2)
# # print(slice2.shape)
# # s = rotationWedge(slice2)

# R = M[0:3, 0:3]
# p = M[0:3, 3]
# temp = rotationWedge(p) * R
# temp = R.row_join(temp)
# temp2 = sf.Matrix.zeros(3,3).row_join(R)
# ret = temp.col_join(temp2)
# print(ret)
# print(-R)
##########
# Using exponential of a symbol

# r1 = sf.Matrix([1,2,3,4]).transpose()
# r2 = sf.Matrix([5, 6, 7, 8]).transpose()
# r3 = sf.Matrix([9, 10, 11, 12]).transpose()

# M = r1.col_join(r2).col_join(r3)
# print(M)

# slice = M[0:3, 0:3]
# print(slice)

# theta = sf.symbols("theta")
# print(slice*theta)
# print(generalExponential(sf.Matrix([1,1,1,1,1,1]), theta))