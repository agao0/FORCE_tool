#!/usr/bin/env python3
####################################################################################################
#
# Generate C++ code using symforce based on manipulator.json
#
# Andrew Gao
# Supervisor: Jianwei Sun
# Made for UCLA Bionics Lab, 2023
#
# Notes:
# - All units are SI unless otherwise specified
#
# Dependencies:
# - python 3.8.17
# - symforce 0.9.0
#
####################################################################################################
# Imports
import symforce

symforce.set_epsilon_to_symbol()

import symforce.symbolic as sf
from symforce.notebook_util import display, display_code, display_code_file, print_expression_tree
import json
from functools import reduce
from symforce.codegen import Codegen, CppConfig
from symforce.values import Values

# Python imports
import argparse
import os

############################################################################################################
############################################### PREMISE ####################################################
############################################################################################################

# For a given manipulator whose parameters are stored in "manipulator_parameters.json", this program will
# output C++ programs using symforce that represent common functions/expressions involved in the manipulator. 
# The key benefit of symforce is that it uses symbolic variables in the expressions to generate C++ code that
# runs fast, which will be useful for RTC (real-time control).

# POSSIBLE FUNCTIONS: the program can be tweaked to produce any of the following:
# forward kinematics map: joint angles vector -> 4x4 matrix
# adjoint map: joint angles vector -> forward kinematics map -> 6x6 matrix
# adjoint inverse map: joint angles vector -> forward kinematics map -> 6x6 matrix
# spatial Jacobian map: joint angles vector -> 6xn matrix
# body Jacobian map: joint angles vector -> 6xn matrix
# Manipulator inertia matrix: joint angles vector -> nxn matrix
# Coriolis matrix: joint angles vector, joint velocities vector -> nxn matrix
# Coriolis and gravity matrix: joint angles vector, joint velocities vector -> nxn matrix

############################################################################################################
############################################ JSON SETUP ####################################################
############################################################################################################
# Note: all data is assumed to be in order from manipulator base to end-effector

# We assume that we are given a robot parameter file, titled "manipulator.json" which contains the 
# following 2 sets of data:
#
#   -> the manipulator's physical data, "manipulator_data", a dictionary with:
#       -> "joint_twists": a list of joint twists in the form ["r", omega, q] for revolute joints and
#           ["p", v] for prismatic joints, defined relative to the inertial frame
#       -> "link_masses": a list of the link masses in order
#       -> "principal_moments": a list of the 3 principal moments of each link in order (XYZ)
#       -> "center_masses": a list of the positions of the center of masses of each link in order
#       -> "principal_axes": a list of the 3 principal axes of each link corresponding to the 3 principal
#           moments (XYZ)
#       
#   -> the toggle "generateDynamics", that will output C++ functions corresponding to the manipulator's
#       inertia matrix, Coriolis matrix, and CoriolisAndGravity combo matrix
#
#   -> points of interest data, "POI_data", a dictionary where each key is the name of the POI, and each
#       value is of the form [N, position, orientation, 0/1, 0/1, 0/1, 0/1] where:
#       -> N is the link number of the POI
#       -> position is the position of the POI relative to the inertial frame of the manipulator
#       -> orientation is the orientation of the POI relative to the inertial frame
#       -> each 0/1 toggles the generation of the POI's: Adjoint, AdjointInv, BodyJacobian, and
#          SpatialJacobian in that order
#
# NOTE: Adjoint is the adjoint of the rigid body transformation from the POI frame to the inertial frame,
#
############################################################################################################
############################################ HELPER FUNCTIONS ##############################################
############################################################################################################

def processTwists(lst):
# output twist coordinates associated with each twist in a list of joint twists
    ret = []
    for i in lst:
        if i[0] == "r":
            omega = sf.Matrix(i[1])
            n_omega = -1*omega
            q = sf.Matrix(i[2])
            v = n_omega.cross(q)
            t_coords = v.col_join(omega)
        elif i[0] == "p":
            v = sf.Matrix(i[1])
            t_coords = v.col_join(sf.Matrix31([0, 0, 0]))
        ret += [t_coords]
    return ret

def rotationWedge(vec):
# assume 3 x 1 col vector, either sf or list
    ret = sf.Matrix33()
    ret[0, 1] = -vec[2]
    ret[1, 0] = vec[2]
    ret[0, 2] = vec[1]
    ret[2, 0] = -vec[1]
    ret[2, 1] = vec[0]
    ret[1, 2] = -vec[0]
    return ret

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

def generalAdjoint(g: sf.Matrix44) -> sf.Matrix66:
    R = g[0:3, 0:3]
    p = g[0:3, 3]
    temp = rotationWedge(p) * R
    temp = R.row_join(temp)
    temp2 = sf.Matrix.zeros(3,3).row_join(R)
    ret = temp.col_join(temp2)
    return ret

def generalInverseAdjoint(g: sf.Matrix44) -> sf.Matrix66:
    R = g[0:3, 0:3]
    Rt = R.transpose()
    p = g[0:3, 3]
    temp = -Rt * rotationWedge(p)
    temp = Rt.row_join(temp)
    temp2 = sf.Matrix.zeros(3,3).row_join(Rt)
    ret = temp.col_join(temp2)
    return ret

# Constants
gravity = 9.81

############################################################################################################
############################################# ARGPARSE #####################################################
############################################################################################################

parser = argparse.ArgumentParser(description="Generate C++ programs of common manipulator functions based on a JSON file")
parser.add_argument('manipulator_JSON', help="Specify a manipulator JSON file to process", type=str, nargs=1)
parser.add_argument('-o', '--output_dir', required=False, type=str, nargs=1, help="Specify a folder to output C++ files. Note: all generated files will be located in a subdirectory with name equal to the namespace argument")
parser.add_argument('-n', '--manipulator_name', required=False, type=str, nargs=1, help="Specify a namespace for the C++ file(s). Note: if no namespace given, JSON file name will be used as default namespace")
parser.add_argument('-v', '--verbose', required=False, action='store_true', help="Flags verbose generation.")

args=parser.parse_args()
jsonPath = args.manipulator_JSON[0]
outputDir = args.output_dir
fileName = os.path.basename(jsonPath)
manipulatorName = args.manipulator_name
verbose = args.verbose

assert fileName[-4:].lower() == "json", "Check file type" # must be JSON file

# MANIPULATOR NAME (for function namespaces in the C++ programs):
if manipulatorName:
    M_NAME = manipulatorName[0].strip()
else:
    M_NAME = fileName[0:-5].strip()

# OUTPUT PATH
if outputDir:
    path = os.path.join(os.getcwd(), outputDir[0].strip())
else:
    path = os.path.join(os.getcwd(), "force_output")
try:
    os.mkdir(path)
except FileExistsError:
    if not os.path.isdir(path):
        raise NotADirectoryError("Output directory is not a directory")
    print("\n")
    print("WARNING:")
    print("\'" + path + "\' output directory exists already. This program will replace previously generated C++ files in a subdirectory if they share the same namspace. If a namespace flag -n was used, you may generate C++ files of different namespaces without overriding.\n")
    user_input = input("Continue? [y/n]\n")
    if user_input.lower() == 'yes' or user_input.lower() == 'y':
        print("Continuing...\n")
    else:
        raise FileExistsError("Will not override files")

############################################################################################################
############################################### MAIN #######################################################
############################################################################################################
# Data parsing and processing

with open(jsonPath) as f:
    try:
        # Manipulator Data retrieval
        data = json.load(f)
        manipulator_data = data['manipulator_data']
        joint_twist_data = manipulator_data["joint_twists"]
        link_masses = manipulator_data["link_masses"]
        principal_moments = manipulator_data["principal_moments"]
        center_masses = manipulator_data["center_masses"]
        principal_axes = manipulator_data["principal_axes"]

        # Points of Interest data retrieval
        points_of_interest = data["points_of_interest"]

        # Dynamics Toggle retrieval
        generateDynamics = data["generateDynamics"]
    except: 
        raise Exception("CHECK MANIPULATOR DATA FORMATTING") 

NUM_ANGLES = len(joint_twist_data)

# Wrapper classes to accomodate unknown number of joints
class anglesMatrix(sf.Matrix):
    SHAPE = (NUM_ANGLES, 1)

class JacobianMatrix(sf.Matrix):
    SHAPE = (6, NUM_ANGLES)

joint_twist_coordinates = processTwists(joint_twist_data)


######################################### GENERATE DYNAMICS ###############################################

if generateDynamics:
    class mInertiaMatrix(sf.Matrix):
        SHAPE = (NUM_ANGLES, NUM_ANGLES)

    class velocityMatrix(sf.Matrix):
        SHAPE = (NUM_ANGLES, 1)

    class mCoriolisMatrix(sf.Matrix):
        SHAPE = (NUM_ANGLES, NUM_ANGLES)

    # compute inertia matrices
    inertiaMatrices = []
    for i in range(NUM_ANGLES):
        curr = sf.Matrix.eye(3, 3)
        curr = curr*link_masses[i]
        curr = curr.row_join(sf.Matrix.zeros(3, 3))
        temp = sf.Matrix.zeros(3, 3)
        temp = temp.row_join(sf.Matrix.diag(principal_moments[i]))
        ret = curr.col_join(temp)
        inertiaMatrices += [ret]

    def manipulatorInertiaMatrix(joint_angles: anglesMatrix) -> mInertiaMatrix:
        ret = sf.Matrix.zeros(NUM_ANGLES, NUM_ANGLES)
        for i in range(NUM_ANGLES):
            # Compute body Jacobian of ith link
            position = sf.Matrix(center_masses[i])
            position = position.col_join(sf.Matrix([1]))
            orientation = sf.Matrix(principal_axes[i]).T
            orientation = orientation.col_join(sf.Matrix.zeros(1, 3))
            g_sli0 = orientation.row_join(position)
            tempJacobian = sf.Matrix.zeros(6, NUM_ANGLES)
            for j in range(i + 1):
                expr = [generalExponential(joint_twist_coordinates[k], joint_angles[k]) for k in range(j, i + 1)]
                expr = reduce(lambda x, y: x*y, expr)
                expr = expr*g_sli0
                adjoint = generalInverseAdjoint(expr)
                temp = adjoint*joint_twist_coordinates[j]
                tempJacobian[:,j] = temp
            ret += tempJacobian.T * inertiaMatrices[i] * tempJacobian
        return ret

    codegen = Codegen.function(func=manipulatorInertiaMatrix, name="ManipulatorInertiaMatrix", config=CppConfig())
    codegen_data = codegen.generate_function(output_dir=path, namespace=M_NAME)
    if verbose:
        print("Manipulator inertia matrix generated in {}:".format(codegen_data.output_dir))
        for f in codegen_data.generated_files:
            print("  |- {}\n".format(f.relative_to(codegen_data.output_dir)))

    def coriolisMatrix(joint_angles: anglesMatrix, velocities: velocityMatrix) -> mCoriolisMatrix:
        mInertias = manipulatorInertiaMatrix(joint_angles)
        ret = sf.Matrix.zeros(NUM_ANGLES, NUM_ANGLES)
        for i in range(NUM_ANGLES):
            for j in range(NUM_ANGLES):
                sum = 0
                for k in range(NUM_ANGLES):
                    sum += (mInertias[i, j].diff(joint_angles[k]) + mInertias[i, k].diff(joint_angles[j]) - mInertias[k, j].diff(joint_angles[i]))*velocities[k]
                ret[i,j] = 0.5*sum
        return ret

    codegen = Codegen.function(func=coriolisMatrix, name="CoriolisMatrix", config=CppConfig())
    codegen_data = codegen.generate_function(output_dir=path, namespace=M_NAME)
    print("Coriolis matrix generated in {}:".format(codegen_data.output_dir))
    if verbose:
        for f in codegen_data.generated_files:
            print("  |- {}\n".format(f.relative_to(codegen_data.output_dir)))

    def coriolisAndGravity(joint_angles: anglesMatrix, velocities: velocityMatrix) -> anglesMatrix:
        h = []
        for i in range(NUM_ANGLES):
            position = sf.Matrix(center_masses[i])
            position = position.col_join(sf.Matrix([1]))
            orientation = sf.Matrix(principal_axes[i]).T
            orientation = orientation.col_join(sf.Matrix.zeros(1, 3))
            g_sli0 = orientation.row_join(position)
            expr = [generalExponential(joint_twist_coordinates[k], joint_angles[k]) for k in range(i + 1)]
            expr = reduce(lambda x, y: x*y, expr)
            expr = expr*g_sli0
            h += [expr[2, 3]]
        potentialEnergy = sum([link_masses[i] * gravity * h[i] for i in range(NUM_ANGLES)])
        gravityVector = sf.Matrix([potentialEnergy.diff(joint_angles[k]) for k in range(NUM_ANGLES)])
        coriolis = coriolisMatrix(joint_angles, velocities)
        return coriolis * velocities + gravityVector

    codegen = Codegen.function(func=coriolisAndGravity, name="CoriolisAndGravityMatrix", config=CppConfig())
    codegen_data = codegen.generate_function(output_dir=path, namespace=M_NAME)
    if verbose:
        print("Coriolis and Gravity combo matrix generated in {}:".format(codegen_data.output_dir))
        for f in codegen_data.generated_files:
            print("  |- {}\n".format(f.relative_to(codegen_data.output_dir)))
        

####################################### GENERATE POINTS OF INTEREST ###############################################

for POI in points_of_interest:
    try:
        generateAdjoint = points_of_interest[POI][3]
        generateAdjointInv = points_of_interest[POI][4]
        generateBodyJacobian = points_of_interest[POI][5]
        generateSpatialJacobian = points_of_interest[POI][6]
        generateFKM = points_of_interest[POI][7]
    except:
        raise Exception("CHECK POINTS OF INTEREST TOGGLES FORMATTING") 

    if generateAdjoint:
        def adjointMap(joint_angles: anglesMatrix) -> sf.Matrix66:
            expr = [generalExponential(joint_twist_coordinates[i], joint_angles[i]) for i in range(points_of_interest[POI][0])]
            expr = reduce(lambda x, y: x*y, expr)
            position = sf.Matrix(points_of_interest[POI][1])
            position = position.col_join(sf.Matrix([1]))
            orientation = sf.Matrix(points_of_interest[POI][2])
            orientation = orientation.col_join(sf.Matrix.zeros(1, 3))
            g_st0 = orientation.row_join(position)
            g_sttheta = expr*g_st0
            return generalAdjoint(g_sttheta)

        codegen = Codegen.function(func=adjointMap, name=POI+"Adjoint", config=CppConfig())
        codegen_data = codegen.generate_function(output_dir=path, namespace=M_NAME)
        if verbose:
            print(POI.upper() + " Adjoint Map generated in {}:".format(codegen_data.output_dir))
            for f in codegen_data.generated_files:
                print("  |- {}\n".format(f.relative_to(codegen_data.output_dir)))

    if generateAdjointInv:
        def adjointInvMap(joint_angles: anglesMatrix) -> sf.Matrix66:
            expr = [generalExponential(joint_twist_coordinates[i], joint_angles[i]) for i in range(points_of_interest[POI][0])]
            expr = reduce(lambda x, y: x*y, expr)
            position = sf.Matrix(points_of_interest[POI][1])
            position = position.col_join(sf.Matrix([1]))
            orientation = sf.Matrix(points_of_interest[POI][2])
            orientation = orientation.col_join(sf.Matrix.zeros(1, 3))
            g_st0 = orientation.row_join(position)
            g_sttheta = expr*g_st0
            return generalInverseAdjoint(g_sttheta)

        codegen = Codegen.function(func=adjointInvMap, name=POI+"AdjointInv", config=CppConfig())
        codegen_data = codegen.generate_function(output_dir=path, namespace=M_NAME)
        if verbose:
            print(POI.upper() + " Adjoint Inverse Map generated in {}:".format(codegen_data.output_dir))
            for f in codegen_data.generated_files:
                print("  |- {}\n".format(f.relative_to(codegen_data.output_dir)))

    if generateBodyJacobian:
        def bodyJacobian(joint_angles: anglesMatrix) -> JacobianMatrix:
            position = sf.Matrix(points_of_interest[POI][1])
            position = position.col_join(sf.Matrix([1]))
            orientation = sf.Matrix(points_of_interest[POI][2])
            orientation = orientation.col_join(sf.Matrix.zeros(1, 3))
            g_st0 = orientation.row_join(position)
            tempJacobian = sf.Matrix.zeros(6, NUM_ANGLES)
            for j in range(points_of_interest[POI][0]):
                expr = [generalExponential(joint_twist_coordinates[k], joint_angles[k]) for k in range(j, points_of_interest[POI][0])]
                expr = reduce(lambda x, y: x*y, expr)
                expr = expr*g_st0
                adjoint = generalInverseAdjoint(expr)
                temp = adjoint*joint_twist_coordinates[j]
                tempJacobian[:,j] = temp
            return tempJacobian

        codegen = Codegen.function(func=bodyJacobian, name=POI+"BodyJacobian", config=CppConfig())
        codegen_data = codegen.generate_function(output_dir=path, namespace=M_NAME)
        if verbose:
            print(POI.upper() + " Body Jacobian Map generated in {}:".format(codegen_data.output_dir))
            for f in codegen_data.generated_files:
                print("  |- {}\n".format(f.relative_to(codegen_data.output_dir)))

    if generateSpatialJacobian:
        def spatialJacobian(joint_angles: anglesMatrix) -> JacobianMatrix:
            bodyJac = bodyJacobian(joint_angles)
            adj = adjointMap(joint_angles)
            return adj*bodyJac

        codegen = Codegen.function(func= spatialJacobian, name=POI+"SpatialJacobian", config=CppConfig())
        codegen_data = codegen.generate_function(output_dir=path, namespace=M_NAME)
        if verbose:
            print(POI.upper() + " Spatial Jacobian Map generated in {}:".format(codegen_data.output_dir))
            for f in codegen_data.generated_files:
                print("  |- {}\n".format(f.relative_to(codegen_data.output_dir)))

    if generateFKM:
        def forwardKinematicsMap(joint_angles: anglesMatrix) -> sf.Matrix44:
            expr = [generalExponential(joint_twist_coordinates[i], joint_angles[i]) for i in range(points_of_interest[POI][0])]
            expr = reduce(lambda x, y: x*y, expr)
            position = sf.Matrix(points_of_interest[POI][1])
            position = position.col_join(sf.Matrix([1]))
            orientation = sf.Matrix(points_of_interest[POI][2])
            orientation = orientation.col_join(sf.Matrix.zeros(1, 3))
            g_st0 = orientation.row_join(position)
            g_sttheta = expr*g_st0
            return g_sttheta

        codegen = Codegen.function(func= forwardKinematicsMap, name=POI+"ForwardKinematicsMap", config=CppConfig())
        codegen_data = codegen.generate_function(output_dir=path, namespace=M_NAME)
        if verbose:
            print(POI.upper() + " Forward Kinematics Map generated in {}:".format(codegen_data.output_dir))
            for f in codegen_data.generated_files:
                print("  |- {}\n".format(f.relative_to(codegen_data.output_dir)))