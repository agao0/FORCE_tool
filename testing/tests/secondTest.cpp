#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <fstream>
#include <cstdlib>
#include <ctime>

#include <exoLeftDynamics.hpp>
#include "exoRightDynamics.hpp"
#include "leftLowerSensorAdjointInvMap.h"
#include "leftLowerSensorAdjointMap.h"
#include "leftLowerSensorBodyJacobian.h"
#include "leftUpperSensorAdjointInvMap.h"
#include "leftUpperSensorAdjointMap.h"
#include "leftUpperSensorBodyJacobian.h"
#include "leftWristSensorAdjointInvMap.h"
#include "leftWristSensorAdjointMap.h"
#include "leftWristSensorBodyJacobian.h"
#include "ManipulatorInertiaMatrix.h"
#include "CoriolisAndGravityMatrix.h"

using JointVector = Eigen::Matrix<double, 7, 1>;

float RandomFloat(float a, float b) {
    float random = ((float) rand()) / (float) RAND_MAX;
    float diff = b - a;
    float r = random * diff;
    return a + r;
}

int main(void) {
    float max = 0;
    constexpr double pi = 3.14159265358979323846;

    srand(time(0));
    int NUM_RUNS = 1000;
    float theta_upper = 2*pi;
    // std::ofstream output ("errors_POSTCHANGE.txt");
    
    for(int j = 2; j < 13; j++) {
        float thetad_upper = j;
        std::cout << "THETAD_UPPER: " << thetad_upper << std::endl;

    for (int i = 0; i < NUM_RUNS; i++) {
        std::cout << "Run: " << i << std::endl;

        float theta1 = RandomFloat(0.0, theta_upper);
        float theta2 = RandomFloat(0.0, theta_upper);
        float theta3 = RandomFloat(0.0, theta_upper);
        float theta4 = RandomFloat(0.0, theta_upper);
        float theta5 = RandomFloat(0.0, theta_upper);
        float theta6 = RandomFloat(0.0, theta_upper);
        float theta7 = RandomFloat(0.0, theta_upper);

        float thetad1 = RandomFloat(-thetad_upper, thetad_upper);
        float thetad2 = RandomFloat(-thetad_upper, thetad_upper);
        float thetad3 = RandomFloat(-thetad_upper, thetad_upper);
        float thetad4 = RandomFloat(-thetad_upper, thetad_upper);
        float thetad5 = RandomFloat(-thetad_upper, thetad_upper);
        float thetad6 = RandomFloat(-thetad_upper, thetad_upper);
        float thetad7 = RandomFloat(-thetad_upper, thetad_upper);

        // Joint angles/velocities to test
        const JointVector test_angle_vector = (JointVector() << theta1, theta2, theta3, theta4, theta5, theta6, theta7).finished();
        const JointVector test_velocity_vector = (JointVector() << thetad1, thetad2, thetad3, thetad4, thetad5, thetad6, thetad7).finished();

        // LEFT TEST
        const Eigen::Matrix<double, 6, 7> LLBJ = sym::Leftlowersensorbodyjacobian(test_angle_vector);
        const Eigen::Matrix<double, 6, 6> LLAdj = sym::Leftlowersensoradjointmap(test_angle_vector);
        const Eigen::Matrix<double, 6, 6> LLAdjInv = sym::Leftlowersensoradjointinvmap(test_angle_vector);

        const Eigen::Matrix<double, 6, 7> LUBJ = sym::Leftuppersensorbodyjacobian(test_angle_vector);
        const Eigen::Matrix<double, 6, 6> LUAdj = sym::Leftuppersensoradjointmap(test_angle_vector);
        const Eigen::Matrix<double, 6, 6> LUAdjInv = sym::Leftuppersensoradjointinvmap(test_angle_vector);

        const Eigen::Matrix<double, 6, 7> LWBJ = sym::Leftwristsensorbodyjacobian(test_angle_vector);
        const Eigen::Matrix<double, 6, 6> LWAdj = sym::Leftwristsensoradjointmap(test_angle_vector);
        const Eigen::Matrix<double, 6, 6> LWAdjInv = sym::Leftwristsensoradjointinvmap(test_angle_vector);

        const Eigen::Matrix<double, 7, 7> LeftMIM = sym::Manipulatorinertiamatrix(test_angle_vector);
        const Eigen::Matrix<double, 7, 1> LeftCAG = sym::Coriolisandgravitymatrix(test_angle_vector, test_velocity_vector);

        Eigen::Matrix<double, 6, 7, Eigen::RowMajor> sympyLLBJ;
        Eigen::Matrix<double, 6, 7, Eigen::RowMajor> sympyLUBJ;
        Eigen::Matrix<double, 6, 7, Eigen::RowMajor> sympyLWBJ;
        Eigen::Matrix<double, 6, 6, Eigen::RowMajor> sympyLUAdj; // 6x6 for adjUpperT
        Eigen::Matrix<double, 6, 6, Eigen::RowMajor> sympyLLAdj; // 6x6 for adjLowerT
        Eigen::Matrix<double, 6, 6, Eigen::RowMajor> sympyLWAdj; // 6x6 for adjWristT
        Eigen::Matrix<double, 6, 6, Eigen::RowMajor> OTHER4; // 6x6 for coAdjUpperT
        Eigen::Matrix<double, 6, 6, Eigen::RowMajor> OTHER5; // 6x6 for coAdjLowerT
        Eigen::Matrix<double, 6, 6, Eigen::RowMajor> OTHER6; // 6x6 for coAdjWristT
        Eigen::Matrix<double, 6, 6, Eigen::RowMajor> OTHER7; // 6x6 for coAdjUpperTInv
        Eigen::Matrix<double, 6, 6, Eigen::RowMajor> OTHER8; // 6x6 for coAdjLowerTInv
        Eigen::Matrix<double, 6, 6, Eigen::RowMajor> OTHER9; // 6x6 for coAdjWristTInv
        Eigen::Matrix<double, 7, 7, Eigen::RowMajor> sympyLMIM; // 7x7 for inertia
        Eigen::Matrix<double, 7, 1> sympyLCAG; // 7x1 for coriolisAndGravity

        LeftArm::computeSymbolicExpressions(sympyLUBJ.data(), sympyLLBJ.data(), sympyLWBJ.data(), sympyLUAdj.data(), sympyLLAdj.data(), sympyLWAdj.data(), OTHER4.data(), OTHER5.data(), OTHER6.data(), OTHER7.data(), OTHER8.data(), OTHER9.data(), sympyLMIM.data(), sympyLCAG.data(), test_angle_vector.data(), test_velocity_vector.data());

        // Verify the result
        const double LLBJ_error = (LLBJ - sympyLLBJ).norm();
        const double LLAdj_error = (LLAdjInv - sympyLLAdj.transpose()).norm();
        // const double LLAdjInv_error = (LLAdjInv - sympyLLAdjInv).norm();

        const double LUBJ_error = (LUBJ - sympyLUBJ).norm();
        const double LUAdj_error = (LUAdjInv - sympyLUAdj.transpose()).norm();
        // const double LUAdjInv_error = (LLBJ - sympyLLBJ).norm();
    
        const double LWBJ_error = (LWBJ - sympyLWBJ).norm();
        const double LWAdj_error = (LWAdjInv - sympyLWAdj.transpose()).norm();
        // const double LLBJ_error = (LLBJ - sympyLLBJ).norm();

        const double LMIM_error = (LeftMIM - sympyLMIM).norm();
        const double LCAG_error = (LeftCAG - sympyLCAG).norm();

        std::cout << "LLBJ error norm: " << LLBJ_error << std::endl;
        std::cout << "LLAdj error norm: " << LLAdj_error << std::endl;
        std::cout << "LUBJ error norm: " << LUBJ_error << std::endl;
        std::cout << "LUAdj error norm: " << LUAdj_error << std::endl;
        std::cout << "LWBJ error norm: " << LWBJ_error << std::endl;
        std::cout << "LWAdj error norm: " << LWAdj_error << std::endl;
        std::cout << "LInertia error norm: " << LMIM_error << std::endl;
        std::cout << "LCG error norm: " << LCAG_error << std::endl;

        if(LCAG_error > max) {
            max = LCAG_error;
        }
    }
    }
    // output.close();
    if(max > 0.01) {
        return 1;
    }
    return 0;
}