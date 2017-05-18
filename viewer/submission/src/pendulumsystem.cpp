#include "pendulumsystem.h"

#include <cassert>
#include "camera.h"
#include "vertexrecorder.h"

// TODO adjust to number of particles.
const int NUM_PARTICLES = 4;

PendulumSystem::PendulumSystem()
{

    // TODO 4.2 Add particles for simple pendulum
    // TODO 4.3 Extend to multiple particles

    // To add a bit of randomness, use e.g.
    // float f = rand_uniform(-0.5f, 0.5f);
    // in your initial conditions.
    
    //even indices are positions, odd indicies are velocites. Particle i's posisition is at index 2i
    
    float f = rand_uniform(0.0f, 0.5f);
    //fixed point
    m_vVecState.push_back(Vector3f(0, 0, 0));
    m_vVecState.push_back(Vector3f(0, 0, 0));
    
    //all others
    Spring *s;
    for (int i = 1; i < NUM_PARTICLES; i++) {
        s = new Spring;
        s->particle1 = i-1;
        s->particle2 = i;
        s->restLength = 0.2f;
        s->stiffness = 1.0f;
        springs.push_back(s);
        m_vVecState.push_back(Vector3f(i+f, 0, 0));
        m_vVecState.push_back(Vector3f(0, 0, 0));
    }
}

//returns vector of integers corresponding to all springs connected to particle i
std::vector<Spring*> PendulumSystem::checkSpringsAtParticle(int i) {
    std::vector<Spring*> connectedSprings;
    for (struct Spring *s : springs) {
        if (s->particle1 == i || s->particle2 == i) {
            connectedSprings.push_back(s);
        }
    }
    return connectedSprings;
}

void PendulumSystem::displaySpringsAtParticle(int i) {
    std::vector<Spring*> connectedSprings = checkSpringsAtParticle(i);
    
    //TODO implement this
}


std::vector<Vector3f> PendulumSystem::evalF(std::vector<Vector3f> state)
{
    std::vector<Vector3f> f(state.size());
    // TODO 4.1: implement evalF
    //  - gravity
    //  - viscous drag
    //  - springs
    
    //for fixed point, set v and f to zero
    f.push_back(Vector3f(0, 0, 0));
    f.push_back(Vector3f(0, 0, 0));
    
    Vector3f v, F, d; //velocity, net force, and distance
    int otherParticle;
    std::vector<Spring*> connectedSprings;
    //printf("%d\n", state.size());
    int particleNum = 1; //particle 0 is fixed point, so ignore it
    for (int i = 2; i < state.size(); i+=2) {
        v = state[i+1];
        F = Vector3f(0, 0, 0);
        connectedSprings = checkSpringsAtParticle(particleNum);
        for (struct Spring *s : connectedSprings) {
            if (s->particle1 == particleNum) {
                otherParticle = s->particle2;
            } else {
                otherParticle = s->particle1;
            }
            d = m_vVecState[i] - m_vVecState[2*otherParticle];
            F = F + -s->stiffness*(d.abs() - s->restLength)*d.normalized();
        }
        F = F + gravitationalForce;
        F = F - dragConstant * v;
        f[i] = v;
        f[i+1] = F;
        particleNum++;
    }
    return f;
}

// render the system (ie draw the particles)
void PendulumSystem::draw(GLProgram& gl)
{
    const Vector3f PENDULUM_COLOR(0.73f, 0.0f, 0.83f);
    gl.updateMaterial(PENDULUM_COLOR);

    // TODO 4.2, 4.3

    for (int i = 0; i < PendulumSystem::getState().size(); i+=2) {
        Vector3f pos = PendulumSystem::getState()[i]; //YOUR PARTICLE POSITION
        //pos.print();
        gl.updateModelMatrix(Matrix4f::translation(pos));
        drawSphere(0.075f, 10, 10);
    }
}
