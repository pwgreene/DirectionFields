#ifndef GELATINSYSTEM_H
#define GELATINSYSTEM_H

#include <vector>

#include "particlesystem.h"



class GelatinSystem : public ParticleSystem
{
    
public:
    GelatinSystem();
    
    // evalF is called by the integrator at least once per time step
    std::vector<Vector3f> evalF(std::vector<Vector3f> state) override;
    
    std::vector<Spring*> springs;
    
    std::vector<Spring*> checkSpringsAtParticle(int i);
    
    Spring* tryMakingSpring(int from_x, int from_y, int from_z, int to_x, int to_y, int to_z, float stiffness, float restLength);
    
    int getIndexOf(int i, int j, int k);
    
    bool isOnCloth(int i, int j, int k);
    
    // draw is called once per frame
    void draw(GLProgram& ctx);
    
    // inherits
    // std::vector<Vector3f> m_vVecState;
private:
    Vector3f gravitationalForce = Vector3f(0, -0.005f, 0);
    
    float distance_between = 0.4f;
    
    float dragConstant = 0.1f;
    
    float structural_stiffness = 2.0f;
    float structural_length = distance_between;
    
    float shear_stiffness = 0.2f;
    float shear_length = distance_between*1.732f;
    
    float flex_stiffness = 1.5f;
    float flex_length = 2*distance_between;
};


#endif
