#ifndef CLOTHSYSTEM_H
#define CLOTHSYSTEM_H

#include <vector>

#include "particlesystem.h"



class ClothSystem : public ParticleSystem
{
    
public:
    ClothSystem();

    // evalF is called by the integrator at least once per time step
    std::vector<Vector3f> evalF(std::vector<Vector3f> state) override;
    
    std::vector<Spring*> springs;
    
    std::vector<Spring*> checkSpringsAtParticle(int i);
    
    Spring* tryMakingSpring(int from_x, int from_y, int to_x, int to_j, float stiffness, float restLength);
    
    int getIndexOf(int i, int j);
    
    bool isOnCloth(int i, int j);

    // draw is called once per frame
    void draw(GLProgram& ctx);

    // inherits
    // std::vector<Vector3f> m_vVecState;
private:
    Vector3f gravitationalForce = Vector3f(0, -0.02f, 0);
    
    float distance_between = 0.3f;
    
    float dragConstant = 0.1f;
    
    float structural_stiffness = 3.0f;
    float structural_length = distance_between;
    
    float shear_stiffness = 2.5f;
    float shear_length = distance_between*1.41f;
    
    float flex_stiffness = 0.8f;
    float flex_length = 2*distance_between;
};


#endif
