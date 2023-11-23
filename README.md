# **Fast Optimal Robotics Codegen-Ermine** (FORCE)
FORCE is a **symforce-based tool** that is used to generate C++ code of common manipulator expressions/functions. The key benefits of symforce are that it uses symbolic variables in its processing to generate C++ code that is computationally efficient, which will be useful for real-time control, and that its code-generation is done relatively fast.

Developed for the Bionics lab at UCLA. 

## Features
The following functions can be generated based on a manipulator's properties from its JSON file. Each function will take in an n-vector of the manipulator's joint angles (as well as an n-vector of the joint velocities if necessary) and output its corresponding matrix:
- **Forward kinematics map**
- **Adjoint map**
- **Adjoint inverse map**
- **Spatial Jacobian map**
- **Body Jacobian map**
- **Manipulator inertia matrix**
- **Coriolis matrix**
- **Coriolis and gravity matrix**

## Installation
FORCE requires two dependencies to run properly:
- [Python](https://www.python.org/) - version 3.8.17 and above
- [Symforce](https://symforce.org/) - version 0.9.0 and above

The output files will require [Eigen](https://eigen.tuxfamily.org/index.php?title=Main_Page) to build and run.

## Usage
The manipulator's parameters must be stored in a JSON file and formatted appropriately. See [**JSON formatting**](#json-formatting) for how to do this explicitly. Once there, `force_tool.py` is used from the command-line directly:

```
python force_tool.py MANIPULATOR.json
```

In addition, the following flags may be toggled:

```sh
python force_tool.py MANIPULATOR.json -o OUTPUT_DIR -n NAMESPACE -v
```
- `-o OUTPUT_DIR` specifies a directory to output the generated C++ files. By default, a directory named `force_output` will be created.
- `-n NAMESPACE` specifies a namespace for the generated C++ files. By default, the JSON file name will be used.
- `-v` toggles verbose generation

Based on your OS and PATH specifications, the `python3` command may be required instead.

## JSON Formatting
> [!NOTE]
> All data is assumed to be in order from manipulator base to end-effector and defined relative to the inertial frame (if applicable)

The robot parameter JSON file must contain the following 3 sets of key-value pairs:
- **manipulator_data**: the manipulator's physical data, stored as a dictionary with the key-value pairs:
  - **joint_twists**: a list of joint twists in the form `["r", omega, q]` for revolute joints and `["p", v]` for prismatic joints
  - **link_masses**: a list of the link masses
  - **principal_moments**: a list of the 3 principal moments of each link
  - **center_masses**: a list of the positions of the center of masses of each link
  - **principal_axes**: a list of the 3 principal axes of each link corresponding to the 3 principal moments
- **generateDynamics**: a 0/1 toggle to generate the _manipulator inertia matrix_, _Coriolis matrix_, and _CoriolisAndGravity_ combo matrix
- **POI_data**: points of interest data, stored as a dictionary where each key is the name of the POI, and each value is of the form `[N, position, orientation, 0/1, 0/1, 0/1, 0/1, 0/1]` with:
  - **N**: the link number of the POI
  - **position**: the position of the POI
  - **orientation**: the orientation of the POI
  - each 0/1 toggles the generation of the POI's: _Adjoint_, _AdjointInv_, _BodyJacobian_, _SpatialJacobian_, and _ForwardKinematicsMap_ ***in that order***

> [!NOTE]
> Adjoint is the adjoint of the rigid body transformation that converts coordinates from the POI frame to the inertial frame. The rest are defined accordingly.

Rather than creating a JSON file from scratch, it is recommended to modify the existing `example.json` file in the top-level directory.

## Testing
A comparison of this tool versus a previous tool that was built using sympy was performed. The results are stored in a Jupyter notebook within the `testing` folder, based on `errors.txt`. Similarly, a skeleton version of the testing code is stored within `testing/tests`, and can be modified or ran using CMake.
