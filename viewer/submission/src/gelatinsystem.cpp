#include "camera.h"
#include "vertexrecorder.h"
#include "gelatinsystem.h"

// your system should at least contain 8x8 particles.
const int W = 4;
const int H = 4;
const int L = 4;
const int NUM_PARTICLES = W*H*L;

GelatinSystem::GelatinSystem()
{
    // TODO 5. Initialize m_vVecState with cloth particles.
    // You can again use rand_uniform(lo, hi) to make things a bit more interesting
    
    Vector3f topLeft = Vector3f(0, 2.5, 0);
    
    m_vVecState.push_back(topLeft);
    m_vVecState.push_back(Vector3f(0, 0, 0));
    
    //all others
    Spring *s;
    int count = 0;
    for (int i = 0; i < H; i++) { //rows
        for (int j = 0; j < W; j++) { //columns
            for (int k = 0; k < L; k++) { //
                //structural springs, only add right, down, and back to avoid duplicates
                s = tryMakingSpring(i, j, k, i, j+1, k, structural_stiffness, structural_length);
                if (s != NULL) {
                    springs.push_back(s);
                }
                s = tryMakingSpring(i, j, k, i+1, j, k, structural_stiffness, structural_length);
                if (s != NULL) {
                    springs.push_back(s);
                }
                s = tryMakingSpring(i, j, k, i, j, k+1, structural_stiffness, structural_length);
                //shear springs, only add downwards (forward and backwards) diagonals
                s = tryMakingSpring(i, j, k, i+1, j+1, k+1, shear_stiffness, shear_length);
                if (s != NULL) {
                    springs.push_back(s);
                }
                s = tryMakingSpring(i, j, k, i+1, j-1, k+1, shear_stiffness, shear_length);
                if (s != NULL) {
                    springs.push_back(s);
                }
                s = tryMakingSpring(i, j, k, i+1, j-1, k-1, shear_stiffness, shear_length);
                if (s != NULL) {
                    springs.push_back(s);
                }
                s = tryMakingSpring(i, j, k, i+1, j+1, k+1, shear_stiffness, shear_length);
                if (s != NULL) {
                    springs.push_back(s);
                }
                s = tryMakingSpring(i, j, k, i+1, j, k+1, shear_stiffness, shear_length);
                if (s != NULL) {
                    springs.push_back(s);
                }
                s = tryMakingSpring(i, j, k, i+1, j, k-1, shear_stiffness, shear_length);
                if (s != NULL) {
                    springs.push_back(s);
                }
                s = tryMakingSpring(i, j, k, i+1, j+1, k, shear_stiffness, shear_length);
                if (s != NULL) {
                    springs.push_back(s);
                }
                s = tryMakingSpring(i, j, k, i+1, j-1, k, shear_stiffness, shear_length);
                if (s != NULL) {
                    springs.push_back(s);
                }
                //flex springs, only add right, down, and back
                s = tryMakingSpring(i, j, k, i, j+2, k, flex_stiffness, flex_length);
                if (s != NULL) {
                    springs.push_back(s);
                }
                s = tryMakingSpring(i, j, k, i+2, j, k, flex_stiffness, flex_length);
                if (s != NULL) {
                    springs.push_back(s);
                }
                s = tryMakingSpring(i, j, k, i, j, k+2, flex_stiffness, flex_length);
                if (s != NULL) {
                    springs.push_back(s);
                }
                if (!(i == 0 && j == 0 && k == 0)) {
                    m_vVecState.push_back(topLeft + Vector3f(j*distance_between, -i*distance_between, k*distance_between));
                    m_vVecState.push_back(Vector3f(0, 0, 0));
                }
                if (i == H-1) {
                    printf("%d\n", count);
                }
                count ++;
                
            }
        }
    }
    
}

std::vector<Spring*> GelatinSystem::checkSpringsAtParticle(int i) {
    std::vector<Spring*> connectedSprings;
    for (struct Spring *s : springs) {
        if (s->particle1 == i || s->particle2 == i) {
            connectedSprings.push_back(s);
        }
    }
    return connectedSprings;
}

Spring* GelatinSystem::tryMakingSpring(int from_x, int from_y, int from_z, int to_x, int to_y, int to_z, float stiffness, float restLength)
{
    Spring *s = new Spring;
    if (isOnCloth(to_x, to_y, to_z)) {
        printf("spring from (%d, %d, %d) to (%d, %d, %d)\n", from_x, from_y, from_z, to_x, to_y, to_z);
        s->particle1 = getIndexOf(from_x, from_y, from_z);
        s->particle2 = getIndexOf(to_x, to_y, to_z);
        s->stiffness = stiffness;
        s->restLength = restLength;
    } else {
        return NULL;
    }
    return s;
}



std::vector<Vector3f> GelatinSystem::evalF(std::vector<Vector3f> state)
{
    std::vector<Vector3f> f(state.size());
    
    Vector3f v, F, d; //velocity, net force, and distance
    int otherParticle;
    std::vector<Spring*> connectedSprings;
    //printf("%d\n", state.size());
    int particleNum = 0;
    for (int i = 0; i < state.size(); i+=2) {
        v = state[i+1];
        F = Vector3f(0, 0, 0);
        if (i/2 < W*H*L-L*W) { //ignore bottom
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
        } //else {printf("%d\n", i/2);}
        f[i] = v;
        f[i+1] = F;
        particleNum++;
    }
    
    return f;
}

//particle in the 1D state array at row i, column j (zero indexed)
int GelatinSystem::getIndexOf(int i, int j, int k)
{
    return (i * W * H) + (j * H) + k;
}

bool GelatinSystem::isOnCloth(int i, int j, int k)
{
    return i < W && j < H && k < L && i >= 0 && j >= 0 && k >= 0;
}

void GelatinSystem::draw(GLProgram& gl)
{
    //TODO 5: render the system
    //         - ie draw the particles as little spheres
    //         - or draw the springs as little lines or cylinders
    //         - or draw wireframe mesh
    
    const Vector3f CLOTH_COLOR(0.9f, 0.9f, 0.9f);
    gl.updateMaterial(CLOTH_COLOR);
    
    for (int i = 0; i < GelatinSystem::getState().size(); i+=2) {
        Vector3f pos = GelatinSystem::getState()[i];
        gl.updateModelMatrix(Matrix4f::translation(pos));
        drawSphere(0.075f, 10, 10);
    }
    
}

