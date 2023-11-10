####################################################################################################
#
# Generate C++ code using symforce based on manipulator_parameters.json
#
# Andrew Gao
# Supervisor: Jianwei Sun
# Made for UCLA Bionics Lab
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

############################################################################################################
############################################### PREMISE ####################################################
############################################################################################################

# For a given manipulator whose parameters are stored in "manipulator_parameters.json", this program will
# output C++ programs using symforce that represent common functions/expressions involved in the manipulator. 
# The key benefit of symforce is that it uses symbolic variables in the expressions to generate C++ code, 
# which will be useful for RTC (real-time control).

# POSSIBLE FUNCTIONS:
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
# Note: data formatted so that prismatic and revolute joints allowed.

# We assume that we are given a robot parameter file, which possibly contains the following 3 data feeds:
#
#   -> a "simple data" feed, which contains:
#           "simple_joint_twist_data", a list of joint twists 
#           "end_effector_ref_config", the end-effector initial configuration,
#           "simple_points_of_interest", a list of points of interest to which we can compute body 
#                                        jacobians
#      -> this feed is to be used to streamline the process of computing the kinematics/POI of a SINGLE 
#         manipulator
#
#   -> a "POI_data" feed, which contains possibly multiple sets of data, of the form:
#           "DATA_NAME", a prefix for all of its contents
#           "DATA_NAME_joint_twist_data", a list of joint twists
#           "DATA_NAME_points_of_interest", a list of points of interest to which we can compute body 
#                                           jacobians
#      -> this feed is to be used in computing body jacobians to the points of interest using the 
#         respective twist data for SEVERAL manipulators
# 
#   -> a "dynamics_data" data feed, which contains possibly multiple sets of data, of the form:
#           "DATA_NAME", a prefix for all of its contents
#           "DATA_NAME_joint_twist_data", a list of joint twists
#           "DATA_NAME_link_masses", a list of link masses
#           "DATA_NAME_principal_moments", a list of principal moments (XYZ)
#           "DATA_NAME_center_masses", a list of the center of masses of each link in order relative to
#                                      base frame
#           "DATA_NAME_principal_axes", a list of the principal axes corresponding to the principal
#                                       moments (XYZ)

# In addition to the data feeds, the JSON file is also assumed to have the following toggle options to
# generate C++ programs using symforce:
#
#   -> "generateForwardKinematicsMap": generates forward kinematics map based on "simple_data" feed
#   -> "generateSimpleAdjoint": generates adjoint matrix based on "simple_data" feed
#   -> "generateSimpleAdjointInverse": generates inverse adjoint matrix based on "simple_data" feed
#   -> "generateSimpleSpatialJacobian": generates spatial Jacobian matrix based on "simple_data" feed
#   -> "generateSimpleBodyJacobian": generates body Jacobian matrix based on "simple_data" feed
#   -> "generateSimplePOIBodyJacobians": generates body Jacobians matrices based on "simple_data" feed to
#                                        all points of interest
#   -> "generatePOIBodyJacobians": generates all body Jacobians based on "POI_data" feed
#   -> "generateDynamics": generates manipulator inertia matrix, Coriolis matrix, and Coriolis and gravity
#       matrix based on "dynamics_data" feed

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

def CHECK_FORMATTING():
    pass

# Constants
gravity = 9.81

############################################################################################################
############################################### MAIN #######################################################
############################################################################################################

############################################################################################################
########################################## SIMPLE DATA FEED ################################################
############################################################################################################
# "simple_data" feed parsing

try:
    with open('manipulator_parameters.json') as f:
        # Data retrieval
        data = json.load(f)
        simple_data = data['simple_data']
        simple_joint_twist_data = simple_data["simple_joint_twist_data"]
        end_effector_ref_config = sf.Matrix(simple_data['end_effector_ref_config'])
        simple_points_of_interest = simple_data["simple_points_of_interest"]

        # Toggle retrieval
        generateSimpleForwardKinematicsMap = data["generateForwardKinematicsMap"]
        generateSimpleAdjoint = data["generateSimpleAdjoint"]
        generateSimpleAdjointInverse = data["generateSimpleAdjointInverse"]
        generateSimpleSpatialJacobian = data["generateSimpleSpatialJacobian"]
        generateSimpleBodyJacobian = data["generateSimpleBodyJacobian"]
        generateSimplePOIBodyJacobians = data["generateSimplePOIBodyJacobians"]

except: 
    raise Exception("CHECK SIMPLE DATA FORMATTING") 

SIMPLE_NUM_ANGLES = len(simple_joint_twist_data)

# Wrapper classes to accomodate unknown number of joints
class simpleAnglesMatrix(sf.Matrix):
    SHAPE = (SIMPLE_NUM_ANGLES, 1)

class simpleJacobianMatrix(sf.Matrix):
    SHAPE = (6, SIMPLE_NUM_ANGLES)

simple_joint_twist_coordinates = processTwists(simple_joint_twist_data)

# FORWARD KINEMATICS MAP FUNCTION
def forwardKinematicsMap(joint_angles: simpleAnglesMatrix) -> sf.Matrix44:
    expr = [generalExponential(simple_joint_twist_coordinates[i], joint_angles[i]) for i in range(SIMPLE_NUM_ANGLES)]
    expr = reduce(lambda x, y: x*y, expr)
    expr = expr*end_effector_ref_config
    return expr

# Converting forwardKinematicsMap into a function in a C++ file using symforce
if generateSimpleForwardKinematicsMap:
    codegen = Codegen.function(func=forwardKinematicsMap, name="simpleForwardKinematicsMap", config=CppConfig())
    codegen_data = codegen.generate_function(output_dir="/tmp/sf_codegen_forwardKinematicsMap")
    print("Forward Kinematics Map generated in {}:".format(codegen_data.output_dir))
    for f in codegen_data.generated_files:
        print("  |- {}\n".format(f.relative_to(codegen_data.output_dir)))


# ADJOINT MAP FUNCTION
def adjointMap(joint_angles: simpleAnglesMatrix) -> sf.Matrix66:
    expr = forwardKinematicsMap(joint_angles)
    R = expr[0:3, 0:3]
    p = expr[0:3, 3]
    temp = rotationWedge(p) * R
    temp = R.row_join(temp)
    temp2 = sf.Matrix.zeros(3,3).row_join(R)
    ret = temp.col_join(temp2)
    return ret

# Converting adjointMap into a function in a C++ file using symforce
if generateSimpleAdjoint:
    codegen = Codegen.function(func=adjointMap, name="simpleAdjointMap", config=CppConfig())
    codegen_data = codegen.generate_function(output_dir="/tmp/sf_codegen_adjointMap")
    print("Adjoint Map generated in {}:".format(codegen_data.output_dir))
    for f in codegen_data.generated_files:
        print("  |- {}\n".format(f.relative_to(codegen_data.output_dir)))


# INVERSE ADJOINT MAP FUNCTION
def inverseAdjointMap(joint_angles: simpleAnglesMatrix) -> sf.Matrix66:
    expr = forwardKinematicsMap(joint_angles)
    R = expr[0:3, 0:3]
    Rt = R.transpose()
    p = expr[0:3, 3]
    temp = -Rt * rotationWedge(p)
    temp = Rt.row_join(temp)
    temp2 = sf.Matrix.zeros(3,3).row_join(Rt)
    ret = temp.col_join(temp2)
    return ret

# Converting inverseAdjointMap into a function in a C++ file using symforce
if generateSimpleAdjointInverse:
    codegen = Codegen.function(func=inverseAdjointMap, name="simpleAdjointInverseMap", config=CppConfig())
    codegen_data = codegen.generate_function(output_dir="/tmp/sf_codegen_inverseAdjointMap")
    print("Inverse Adjoint Map generated in {}:".format(codegen_data.output_dir))
    for f in codegen_data.generated_files:
        print("  |- {}\n".format(f.relative_to(codegen_data.output_dir)))


# SPATIAL JACOBIAN MAP FUNCTION (J^S_ST(theta))
def spatialJacobianMap(joint_angles: simpleAnglesMatrix) -> simpleJacobianMatrix:
    ret = simple_joint_twist_coordinates[0]
    current_g = sf.Matrix.eye(4, 4)
    for i in range(0, SIMPLE_NUM_ANGLES-1):
        current_g = current_g * generalExponential(simple_joint_twist_coordinates[i], joint_angles[i])
        next_ret = generalAdjoint(current_g)*simple_joint_twist_coordinates[i + 1]
        ret = ret.row_join(next_ret)
    return ret

# Converting spatialJacobianMap into a function in a C++ file using symforce
if generateSimpleSpatialJacobian:
    codegen = Codegen.function(func=spatialJacobianMap, name="simpleSpatialJacobianMap", config=CppConfig())
    codegen_data = codegen.generate_function(output_dir="/tmp/sf_codegen_SpatialJacobianMap")
    print("Spatial Jacobian Map generated in {}:".format(codegen_data.output_dir))
    for f in codegen_data.generated_files:
        print("  |- {}\n".format(f.relative_to(codegen_data.output_dir)))

# BODY JACOBIAN MAP FUNCTION (J^B_ST(theta))
def bodyJacobianMap(joint_angles: simpleAnglesMatrix) -> simpleJacobianMatrix:
    temp = spatialJacobianMap(joint_angles)
    FKM = forwardKinematicsMap(joint_angles)
    adjoint = generalInverseAdjoint(FKM)
    return adjoint*temp

# Converting bodyJacobianMap into a function in a C++ file using symforce
if generateSimpleBodyJacobian:
    codegen = Codegen.function(func=bodyJacobianMap, name="simpleBodyJacobianMap", config=CppConfig())
    codegen_data = codegen.generate_function(output_dir="/tmp/sf_codegen_BodyJacobianMap")
    print("Body Jacobian Map generated in {}:".format(codegen_data.output_dir))
    for f in codegen_data.generated_files:
        print("  |- {}\n".format(f.relative_to(codegen_data.output_dir)))

# Assumes the POI have a position and that they have rotation matrix identity from the base (inertial frame)
if generateSimplePOIBodyJacobians:
    for key in simple_points_of_interest:
        def BodyJacobian(joint_angles: simpleAnglesMatrix) -> simpleJacobianMatrix:
            current = simple_points_of_interest[key]
            position = sf.Matrix(current[1])
            position = position.col_join(sf.Matrix([1]))
            identity = sf.Matrix.eye(3)
            identity = identity.col_join(sf.Matrix.zeros(1, 3))
            g_st0 = identity.row_join(position)
            tempJacobian = sf.Matrix(6, SIMPLE_NUM_ANGLES)
            for j in range(current[0]):
                expr = [generalExponential(simple_joint_twist_coordinates[k], joint_angles[k]) for k in range(j, current[0])]
                expr = reduce(lambda x, y: x*y, expr)
                expr = expr*g_st0
                adjoint = generalInverseAdjoint(expr)
                temp = adjoint*simple_joint_twist_coordinates[j]
                tempJacobian[:,j] = temp
            return tempJacobian

        codegen = Codegen.function(func=BodyJacobian, name=key+"BodyJacobianMap", config=CppConfig())
        codegen_data = codegen.generate_function(output_dir="/tmp/sf_codegen_" + key + "_Body_Jacobian")
        print(key + " Body Jacobian for the POI generated in {}:".format(codegen_data.output_dir))
        for f in codegen_data.generated_files:
            print("  |- {}\n".format(f.relative_to(codegen_data.output_dir)))


############################################################################################################
########################################## POI DATA FEED ###################################################
############################################################################################################
# "POI_data" parsing

try:
    with open('manipulator_parameters.json') as f:
        # Data retrieval
        data = json.load(f)
        POI_data = data["POI_data"]

        # Toggle retrieval
        generatePOIBodyJacobians = data["generatePOIBodyJacobians"]

except: 
    raise Exception("CHECK POI DATA FORMATTING") 

# Assumes the sensors have a position and that they have rotation matrix identity from the base (inertial frame)
if generatePOIBodyJacobians:
    for DATA_NAME in POI_data:
        joint_twist_coordinates = processTwists(POI_data[DATA_NAME][DATA_NAME + "_joint_twist_data"])
        NUM_ANGLES = len(joint_twist_coordinates)

        class anglesMatrix(sf.Matrix):
            SHAPE = (NUM_ANGLES, 1)
        class JacobianMatrix(sf.Matrix):
            SHAPE = (6, NUM_ANGLES)

        pointsOfInterest = POI_data[DATA_NAME][DATA_NAME + "_points_of_interest"]
        for key in pointsOfInterest:
            def BodyJacobian(joint_angles: anglesMatrix) -> JacobianMatrix:
                current = pointsOfInterest[key]
                position = sf.Matrix(current[1])
                position = position.col_join(sf.Matrix([1]))
                identity = sf.Matrix.eye(3)
                identity = identity.col_join(sf.Matrix.zeros(1, 3))
                g_st0 = identity.row_join(position)
                tempJacobian = sf.Matrix.zeros(6, NUM_ANGLES)
                for j in range(current[0]):
                    expr = [generalExponential(joint_twist_coordinates[k], joint_angles[k]) for k in range(j, current[0])]
                    expr = reduce(lambda x, y: x*y, expr)
                    expr = expr*g_st0
                    adjoint = generalInverseAdjoint(expr)
                    temp = adjoint*joint_twist_coordinates[j]
                    tempJacobian[:,j] = temp
                return tempJacobian

            codegen = Codegen.function(func=BodyJacobian, name=key+"BodyJacobian", config=CppConfig())
            codegen_data = codegen.generate_function(output_dir="/tmp/sf_codegen_" + key + "_Body_Jacobian")
            print(key + " Body Jacobian for the POI generated in {}:".format(codegen_data.output_dir))
            for f in codegen_data.generated_files:
                print("  |- {}\n".format(f.relative_to(codegen_data.output_dir)))


############################################################################################################
######################################### DYNAMICS DATA FEED ###############################################
############################################################################################################
# "dynamics_data" parsing

try:
    with open('manipulator_parameters.json') as f:
        # Data retrieval
        dynamics_data = data["dynamics_data"]

        # Toggle retrieval
        generateDynamics = data["generateDynamics"]
except: 
    raise Exception("CHECK DYNAMICS DATA FORMATTING") 

if generateDynamics:
    for DATA_NAME in dynamics_data:
        joint_twist_coordinates = processTwists(dynamics_data[DATA_NAME][DATA_NAME + "_joint_twist_data"])
        link_masses = dynamics_data[DATA_NAME][DATA_NAME + "_link_masses"]
        principal_moments = dynamics_data[DATA_NAME][DATA_NAME + "_principal_moments"]
        center_masses = dynamics_data[DATA_NAME][DATA_NAME + "_center_masses"]
        principal_axes = dynamics_data[DATA_NAME][DATA_NAME + "_principal_axes"]

        NUM_ANGLES = len(joint_twist_coordinates)

        class anglesMatrix(sf.Matrix):
            SHAPE = (NUM_ANGLES, 1)

        class mInertiaMatrix(sf.Matrix):
            SHAPE = (NUM_ANGLES, NUM_ANGLES)

        class velocityMatrix(sf.Matrix):
            SHAPE = (NUM_ANGLES, 1)

        class mCoriolisMatrix(sf.Matrix):
            SHAPE = (NUM_ANGLES, NUM_ANGLES)

        # compute left inertia matrices
        inertiaMatrices = []
        for i in range(NUM_ANGLES):
            curr = sf.Matrix.eye(3, 3)
            curr = curr*link_masses[i]
            curr = curr.row_join(sf.Matrix.zeros(3, 3))
            temp = sf.Matrix.zeros(3, 3)
            temp = temp.row_join(sf.Matrix.diag(principal_moments[i]))
            ret = curr.col_join(temp)
            inertiaMatrices += [ret]
            # print(ret)

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

        codegen = Codegen.function(func=manipulatorInertiaMatrix, name=DATA_NAME+"ManipulatorInertiaMatrix", config=CppConfig())
        codegen_data = codegen.generate_function(output_dir="/tmp/sf_codegen_" + DATA_NAME + "ManipulatorInertiaMatrix")
        print(DATA_NAME.upper() + " Manipulator inertia matrix generated in {}:".format(codegen_data.output_dir))
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

        codegen = Codegen.function(func=coriolisMatrix, name=DATA_NAME+"CoriolisMatrix", config=CppConfig())
        codegen_data = codegen.generate_function(output_dir="/tmp/sf_codegen_" + DATA_NAME + "CoriolisMatrix")
        print(DATA_NAME.upper() + " Coriolis matrix generated in {}:".format(codegen_data.output_dir))
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

        codegen = Codegen.function(func=coriolisAndGravity, name=DATA_NAME+"CoriolisAndGravity", config=CppConfig())
        codegen_data = codegen.generate_function(output_dir="/tmp/sf_codegen_" + DATA_NAME + "CoriolisAndGravity")
        print(DATA_NAME.upper() + " Coriolis and Gravity combo matrix generated in {}:".format(codegen_data.output_dir))
        for f in codegen_data.generated_files:
            print("  |- {}\n".format(f.relative_to(codegen_data.output_dir)))
            