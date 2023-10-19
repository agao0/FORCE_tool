#include<iostream>
#include<vector>
#include<Eigen/Dense>
#include <nlohmann/json.hpp>
#include <fstream>

#include "include/exoLeftDynamics.hpp"
#include "include/exoRightDynamics.hpp"
#include "include/LLowerSensorBodyJacobian.h"
#include "include/LUpperSensorBodyJacobian.h"
#include "include/LWristSensorBodyJacobian.h"
#include "include/RLowerSensorBodyJacobian.h"
#include "include/RUpperSensorBodyJacobian.h"
#include "include/RWristSensorBodyJacobian.h"

using JointVector = Eigen::Matrix<double, 7, 1>;
using json = nlohmann::json;

int main(void) {
    // std::ifstream f("test3.json");
    // json data = json::parse(f);
    // std::cout << data["joint_twist_data"] << std::endl;

    // Joint angles to test
    const std::vector<JointVector> test_angles = {
        (JointVector() << 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2).finished(),
        (JointVector() << 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0).finished(),
        (JointVector() << -0.1, -0.2, 0.3, 0.4, -0.2, -0.1, 0.0).finished(),
    };

    for (const JointVector& jointVector: test_angles) {
        const Eigen::Matrix<double, 6, 7> LLBJ = sym::Llowersensorbodyjacobian(jointVector);
        const Eigen::Matrix<double, 6, 7> LUBJ = sym::Luppersensorbodyjacobian(jointVector);
        const Eigen::Matrix<double, 6, 7> LWBJ = sym::Lwristsensorbodyjacobian(jointVector);
        const Eigen::Matrix<double, 6, 7> RLBJ = sym::Rlowersensorbodyjacobian(jointVector);
        const Eigen::Matrix<double, 6, 7> RUBJ = sym::Ruppersensorbodyjacobian(jointVector);
        const Eigen::Matrix<double, 6, 7> RWBJ = sym::Rwristsensorbodyjacobian(jointVector);
        
        // const JointVector otherJointVector = jointVector;

        Eigen::Matrix<double, 6, 7, Eigen::RowMajor> sympyLLBJ;
        Eigen::Matrix<double, 6, 7, Eigen::RowMajor> sympyLUBJ;
        Eigen::Matrix<double, 6, 7, Eigen::RowMajor> sympyLWBJ;
        Eigen::Matrix<double, 6, 7, Eigen::RowMajor> sympyRLBJ;
        Eigen::Matrix<double, 6, 7, Eigen::RowMajor> sympyRUBJ;
        Eigen::Matrix<double, 6, 7, Eigen::RowMajor> sympyRWBJ;
        Eigen::Matrix<double, 6, 6, Eigen::RowMajor> OTHER; // 6x6 for adjUpperT
        Eigen::Matrix<double, 6, 6, Eigen::RowMajor> OTHER2; // 6x6 for adjLowerT
        Eigen::Matrix<double, 6, 6, Eigen::RowMajor> OTHER3; // 6x6 for adjWristT
        Eigen::Matrix<double, 6, 6, Eigen::RowMajor> OTHER4; // 6x6 for coAdjUpperT
        Eigen::Matrix<double, 6, 6, Eigen::RowMajor> OTHER5; // 6x6 for coAdjLowerT
        Eigen::Matrix<double, 6, 6, Eigen::RowMajor> OTHER6; // 6x6 for coAdjWristT
        Eigen::Matrix<double, 6, 6, Eigen::RowMajor> OTHER7; // 6x6 for coAdjUpperTInv
        Eigen::Matrix<double, 6, 6, Eigen::RowMajor> OTHER8; // 6x6 for coAdjLowerTInv
        Eigen::Matrix<double, 6, 6, Eigen::RowMajor> OTHER9; // 6x6 for coAdjWristTInv
        Eigen::Matrix<double, 7, 7, Eigen::RowMajor> OTHER10; // 7x7 for inertia
        Eigen::Matrix<double, 7, 1> OTHER11; // 7x1 for coriolisAndGravity
        Eigen::Matrix<double, 7, 1> velocity; // 7x1 for velocity

        LeftArm::computeSymbolicExpressions(sympyLUBJ.data(), sympyLLBJ.data(), sympyLWBJ.data(), OTHER.data(), OTHER2.data(), OTHER3.data(), OTHER4.data(), OTHER5.data(), OTHER6.data(), OTHER7.data(), OTHER8.data(), OTHER9.data(), OTHER10.data(), OTHER11.data(), jointVector.data(), velocity.data());
        RightArm::computeSymbolicExpressions(sympyRUBJ.data(), sympyRLBJ.data(), sympyRWBJ.data(), OTHER.data(), OTHER2.data(), OTHER3.data(), OTHER4.data(), OTHER5.data(), OTHER6.data(), OTHER7.data(), OTHER8.data(), OTHER9.data(), OTHER10.data(), OTHER11.data(), jointVector.data(), velocity.data());

        // Verify the result
        const double LLBJ_error = (LLBJ - sympyLLBJ).norm();
        const double LUBJ_error = (LUBJ - sympyLUBJ).norm();
        const double LWBJ_error = (LWBJ - sympyLWBJ).norm();
        const double RLBJ_error = (RLBJ - sympyRLBJ).norm();
        const double RUBJ_error = (RUBJ - sympyRUBJ).norm();
        const double RWBJ_error = (RWBJ - sympyRWBJ).norm();
        std::cout << "LLBJ error norm: " << LLBJ_error << std::endl;
        std::cout << "LUBJ error norm: " << LUBJ_error << std::endl;
        std::cout << "LWBJ error norm: " << LWBJ_error << std::endl;
        std::cout << "RLBJ error norm: " << RLBJ_error << std::endl;
        std::cout << "RUBJ error norm: " << RUBJ_error << std::endl;
        std::cout << "RWBJ error norm: " << RWBJ_error << std::endl;


    }
    return 0;
}