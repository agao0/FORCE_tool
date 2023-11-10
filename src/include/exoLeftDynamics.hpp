//exoLeftDynamics.hpp generated on 10/16/2023 at 14:29:59

#ifndef EXO_LEFT_DYNAMICS_H
#define EXO_LEFT_DYNAMICS_H

#define _USE_MATH_DEFINES
#include <cmath>

namespace LeftArm {
void computeSymbolicExpressions(double* upperSensorBodyJacobian, double* lowerSensorBodyJacobian, double* wristSensorBodyJacobian, double* adjUpperT, double* adjLowerT, double* adjWristT, double* coAdjUpperT, double* coAdjLowerT, double* coAdjWristT, double* coAdjUpperTInv, double* coAdjLowerTInv, double* coAdjWristTInv, double* inertia, double* coriolisAndGravity, const double* position, const double* velocity);
void computeCollisionPoint1(double* collisionPoint1, const double* position);
void computeCollisionPoint2(double* collisionPoint2, const double* position);
void computeCollisionPoint3(double* collisionPoint3, const double* position);
void computePartialJacobian1(double* partialJacobian1, const double* position, const double* point);
void computePartialJacobian2(double* partialJacobian2, const double* position, const double* point);
void computePartialJacobian3(double* partialJacobian3, const double* position, const double* point);
void computePartialJacobian4(double* partialJacobian4, const double* position, const double* point);
void computePartialJacobian5(double* partialJacobian5, const double* position, const double* point);
void computePartialJacobian6(double* partialJacobian6, const double* position, const double* point);
void computePartialJacobian7(double* partialJacobian7, const double* position, const double* point);
}

#endif