#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <fstream>
#include <cstdlib>
#include <ctime>

#include <exoLeftDynamics.hpp>
#include "exoRightDynamics.hpp"
#include "LeftLowerSensorBodyJacobian.h"
#include "LeftUpperSensorBodyJacobian.h"
#include "LeftWristSensorBodyJacobian.h"
#include "RightLowerSensorBodyJacobian.h"
#include "RightUpperSensorBodyJacobian.h"
#include "RightWristSensorBodyJacobian.h"

#include "leftManipulatorInertiaMatrix.h"
#include "rightManipulatorInertiaMatrix.h"
#include "leftCoriolisAndGravity.h"
#include "rightCoriolisAndGravity.h"


using JointVector = Eigen::Matrix<double, 7, 1>;

float RandomFloat(float a, float b) {
    float random = ((float) rand()) / (float) RAND_MAX;
    float diff = b - a;
    float r = random * diff;
    return a + r;
}

int main(void) {
    constexpr double pi = 3.14159265358979323846;

    srand(time(0));
    int NUM_RUNS = 2000;
    float theta_upper = 2*pi;
    std::ofstream output ("errors.txt");
    
    for(int j = 2; j < 13; j++) {
        float thetad_upper = j;
        output << "THETAD_UPPER: " << thetad_upper << std::endl;

    for (int i = 0; i < NUM_RUNS; i++) {
        output << "Run: " << i << std::endl;

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
        const Eigen::Matrix<double, 6, 7> LUBJ = sym::Leftuppersensorbodyjacobian(test_angle_vector);
        const Eigen::Matrix<double, 6, 7> LWBJ = sym::Leftwristsensorbodyjacobian(test_angle_vector);
        const Eigen::Matrix<double, 7, 7> LeftMIM = sym::Leftmanipulatorinertiamatrix(test_angle_vector);
        const Eigen::Matrix<double, 7, 1> LeftCAG = sym::Leftcoriolisandgravity(test_angle_vector, test_velocity_vector);
        Eigen::Matrix<double, 6, 7, Eigen::RowMajor> sympyLLBJ;
        Eigen::Matrix<double, 6, 7, Eigen::RowMajor> sympyLUBJ;
        Eigen::Matrix<double, 6, 7, Eigen::RowMajor> sympyLWBJ;
        Eigen::Matrix<double, 6, 6, Eigen::RowMajor> OTHER; // 6x6 for adjUpperT
        Eigen::Matrix<double, 6, 6, Eigen::RowMajor> OTHER2; // 6x6 for adjLowerT
        Eigen::Matrix<double, 6, 6, Eigen::RowMajor> OTHER3; // 6x6 for adjWristT
        Eigen::Matrix<double, 6, 6, Eigen::RowMajor> OTHER4; // 6x6 for coAdjUpperT
        Eigen::Matrix<double, 6, 6, Eigen::RowMajor> OTHER5; // 6x6 for coAdjLowerT
        Eigen::Matrix<double, 6, 6, Eigen::RowMajor> OTHER6; // 6x6 for coAdjWristT
        Eigen::Matrix<double, 6, 6, Eigen::RowMajor> OTHER7; // 6x6 for coAdjUpperTInv
        Eigen::Matrix<double, 6, 6, Eigen::RowMajor> OTHER8; // 6x6 for coAdjLowerTInv
        Eigen::Matrix<double, 6, 6, Eigen::RowMajor> OTHER9; // 6x6 for coAdjWristTInv
        Eigen::Matrix<double, 7, 7, Eigen::RowMajor> sympyLMIM; // 7x7 for inertia
        Eigen::Matrix<double, 7, 1> sympyLCAG; // 7x1 for coriolisAndGravity

        LeftArm::computeSymbolicExpressions(sympyLUBJ.data(), sympyLLBJ.data(), sympyLWBJ.data(), OTHER.data(), OTHER2.data(), OTHER3.data(), OTHER4.data(), OTHER5.data(), OTHER6.data(), OTHER7.data(), OTHER8.data(), OTHER9.data(), sympyLMIM.data(), sympyLCAG.data(), test_angle_vector.data(), test_velocity_vector.data());

        // Verify the result
        const double LLBJ_error = (LLBJ - sympyLLBJ).norm();
        const double LUBJ_error = (LUBJ - sympyLUBJ).norm();
        const double LWBJ_error = (LWBJ - sympyLWBJ).norm();
        const double LMIM_error = (LeftMIM - sympyLMIM).norm();
        const double LCAG_error = (LeftCAG - sympyLCAG).norm();

        output << "LLBJ error norm: " << LLBJ_error << std::endl;
        output << "LUBJ error norm: " << LUBJ_error << std::endl;
        output << "LWBJ error norm: " << LWBJ_error << std::endl;
        output << "LInertia error norm: " << LMIM_error << std::endl;
        output << "LCG error norm: " << LCAG_error << std::endl;

        // RIGHT TEST
        const Eigen::Matrix<double, 6, 7> RLBJ = sym::Rightlowersensorbodyjacobian(test_angle_vector);
        const Eigen::Matrix<double, 6, 7> RUBJ = sym::Rightuppersensorbodyjacobian(test_angle_vector);
        const Eigen::Matrix<double, 6, 7> RWBJ = sym::Rightwristsensorbodyjacobian(test_angle_vector);

        const Eigen::Matrix<double, 7, 7> RightMIM = sym::Rightmanipulatorinertiamatrix(test_angle_vector);
        const Eigen::Matrix<double, 7, 1> RightCAG = sym::Rightcoriolisandgravity(test_angle_vector, test_velocity_vector);
        
        Eigen::Matrix<double, 6, 7, Eigen::RowMajor> sympyRLBJ;
        Eigen::Matrix<double, 6, 7, Eigen::RowMajor> sympyRUBJ;
        Eigen::Matrix<double, 6, 7, Eigen::RowMajor> sympyRWBJ;
        Eigen::Matrix<double, 7, 7, Eigen::RowMajor> sympyRMIM; // 7x7 for inertia
        Eigen::Matrix<double, 7, 1> sympyRCAG; // 7x1 for coriolisAndGravity

        RightArm::computeSymbolicExpressions(sympyRUBJ.data(), sympyRLBJ.data(), sympyRWBJ.data(), OTHER.data(), OTHER2.data(), OTHER3.data(), OTHER4.data(), OTHER5.data(), OTHER6.data(), OTHER7.data(), OTHER8.data(), OTHER9.data(), sympyRMIM.data(), sympyRCAG.data(), test_angle_vector.data(), test_velocity_vector.data());

        // Verify the result
        const double RLBJ_error = (RLBJ - sympyRLBJ).norm();
        const double RUBJ_error = (RUBJ - sympyRUBJ).norm();
        const double RWBJ_error = (RWBJ - sympyRWBJ).norm();
        const double RMIM_error = (RightMIM - sympyRMIM).norm();
        const double RCAG_error = (RightCAG - sympyRCAG).norm();
        output << "RLBJ error norm: " << RLBJ_error << std::endl;
        output << "RUBJ error norm: " << RUBJ_error << std::endl;
        output << "RWBJ error norm: " << RWBJ_error << std::endl;
        output << "RInertia error norm: " << RMIM_error << std::endl;
        output << "RCG error norm: " << RCAG_error << std::endl;
        output << std::endl;
    }
    }
    output.close();

    return 0;
}