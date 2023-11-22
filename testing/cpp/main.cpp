#include <vector>
#include <iostream>

#include <Eigen/Dense>

#include "myFunction.h"
#include "sympyEndEffectorTransformation.hpp"

using JointVector = Eigen::Matrix<double, 7, 1>;

int main(void){
    // Error tolerance 
    constexpr double error_tolerance = 1e-2;

    // Joint angles to test
    const std::vector<JointVector> test_angles = {
        (JointVector() << 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2).finished(),
        (JointVector() << 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0).finished(),
        (JointVector() << -0.1, -0.2, 0.3, 0.4, -0.2, -0.1, 0.0).finished(),
        (JointVector() << 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0).finished()
    };

    // Verify each joint angle
    for(const JointVector& angle : test_angles){
        // Result from SymForce
        const Eigen::Matrix4d symforce = sym::Myfunction(angle);

        // Result from Sympy
        Eigen::Matrix<double, 4, 4, Eigen::RowMajor> sympy;
        computeEndEffectorTransformation(sympy.data(), angle.data());

        // Verify the result
        const double error_norm = (symforce - sympy).norm();
        if(error_norm < error_tolerance){
            std::cout << "Passed with error norm: " << error_norm << std::endl;
        } else{
            std::cout << "Failed with error norm: " << error_norm << std::endl;
        }
    }

    return 0;
}