#ifndef PENDULUMSYSTEM_H
#define PENDULUMSYSTEM_H

#include <vector>

#include "particlesystem.h"


class PendulumSystem : public ParticleSystem
{
public:
    PendulumSystem();
    
    //store which springs are connected to which particles. particle1 of each Spring struct is the one with smaller index in the state, so as to avoid repeats
    std::vector<Spring*> springs;
    
    Vector3f gravitationalForce = Vector3f(0, -0.2f, 0);
    float dragConstant = 0.05f;
    
    std::vector<Spring*> checkSpringsAtParticle(int i);
    void displaySpringsAtParticle(int i);

    std::vector<Vector3f> evalF(std::vector<Vector3f> state) override;
    void draw(GLProgram&);

    // inherits 
    // std::vector<Vector3f> m_vVecState;
};

#endif
