#include "timestepper.h"

#include <cstdio>

void ForwardEuler::takeStep(ParticleSystem* particleSystem, float stepSize)
{
   //X(t+h) = X + h*f(X, t)
    std::vector<Vector3f> X0 = particleSystem->getState();
    std::vector<Vector3f> X0_der = particleSystem->evalF(X0);
    std::vector<Vector3f> X1;
    for (int i = 0; i < X0.size(); i++) {
        X1.push_back(X0[i] + stepSize * X0_der[i]);
    }
    particleSystem->setState(X1);
}

void Trapezoidal::takeStep(ParticleSystem* particleSystem, float stepSize)
{
    //X(t+h) = X + h/2*(f0 + f1)
    int i;
    std::vector<Vector3f> X0 = particleSystem->getState();
    std::vector<Vector3f> X1;
    std::vector<Vector3f> f0 = particleSystem->evalF(X0);
    std::vector<Vector3f> f1;
    for (i = 0; i < f0.size(); i++) {
        f1.push_back(X0[i] + stepSize*f0[i]);
    }
    f1 = particleSystem->evalF(f1);
    for (i = 0; i < X0.size(); i++) {
        X1.push_back(X0[i] + stepSize/2*(f0[i] + f1[i]));
    }
    particleSystem->setState(X1);
}


void RK4::takeStep(ParticleSystem* particleSystem, float stepSize)
{
    int i;
    std::vector<Vector3f> X0 = particleSystem->getState();
    std::vector<Vector3f> X1;
    std::vector<Vector3f> k1 = particleSystem->evalF(X0);
    std::vector<Vector3f> k2, k3, k4;
    //calculate k2
    for (i = 0; i < k1.size(); i++) {
        k2.push_back(X0[i] + stepSize/2*k1[i]);
    }
    k2 = particleSystem->evalF(k2);
    //calculate k3
    for (i = 0; i < k1.size(); i++) {
        k3.push_back(X0[i] + stepSize/2*k2[i]);
    }
    k3 = particleSystem->evalF(k3);
    //calculate k4
    for (i = 0; i < k1.size(); i++) {
        k4.push_back(X0[i] + stepSize*k3[i]);
    }
    k4 = particleSystem->evalF(k4);
    for (i = 0; i < X0.size(); i++) {
        X1.push_back(X0[i] + stepSize/6*(k1[i] + 2*k2[i] + 2*k3[i] + k4[i]));
    }
    particleSystem->setState(X1);
}

