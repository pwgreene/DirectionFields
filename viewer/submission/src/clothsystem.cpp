#include "clothsystem.h"
#include "camera.h"
#include "vertexrecorder.h"

 // your system should at least contain 8x8 particles.
const int W = 8;
const int H = 8;
const int NUM_PARTICLES = W*H;

ClothSystem::ClothSystem()
{
    // TODO 5. Initialize m_vVecState with cloth particles. 
    // You can again use rand_uniform(lo, hi) to make things a bit more interesting
    
    Vector3f topLeft = Vector3f(-4, 4, 0);
    
    m_vVecState.push_back(topLeft);
    m_vVecState.push_back(Vector3f(0, 0, 0));
    
    printf("width %d\n", W*2);
    //all others
    Spring *s;
    for (int i = 0; i < H; i++) { //rows
        for (int j = 0; j < W; j++) { //columns
            //structural springs, only add right and down to avoid duplicates
            s = tryMakingSpring(i, j, i, j+1, structural_stiffness, structural_length);
            if (s != NULL) {
                springs.push_back(s);
            }
            s = tryMakingSpring(i, j, i+1, j, structural_stiffness, structural_length);
            if (s != NULL) {
                springs.push_back(s);
            }
            //shear springs, only add bottom left and right diagonals
            s = tryMakingSpring(i, j, i+1, j+1, shear_stiffness, shear_length);
            if (s != NULL) {
                springs.push_back(s);
            }
            s = tryMakingSpring(i, j, i+1, j-1, shear_stiffness, shear_length);
            if (s != NULL) {
                springs.push_back(s);
            }
            //flex springs, only add right and down
            s = tryMakingSpring(i, j, i, j+2, flex_stiffness, flex_length);
            if (s != NULL) {
                springs.push_back(s);
            }
            s = tryMakingSpring(i, j, i+2, j, flex_stiffness, flex_length);
            if (s != NULL) {
                springs.push_back(s);
            }
            if (!(i == 0 && j == 0)) {
                m_vVecState.push_back(topLeft + Vector3f(j*distance_between, -i*distance_between, rand_uniform(-0.05f, 0.05f)));
                m_vVecState.push_back(Vector3f(0, 0, 0));
            }
        }
    }
    
}

std::vector<Spring*> ClothSystem::checkSpringsAtParticle(int i) {
    std::vector<Spring*> connectedSprings;
    for (struct Spring *s : springs) {
        if (s->particle1 == i || s->particle2 == i) {
            connectedSprings.push_back(s);
        }
    }
    return connectedSprings;
}
    
Spring* ClothSystem::tryMakingSpring(int from_x, int from_y, int to_x, int to_y, float stiffness, float restLength)
{
    Spring *s = new Spring;
    if (isOnCloth(to_x, to_y)) {
        //printf("spring from (%d, %d) to (%d, %d)\n", from_x, from_y, to_x, to_y);
        s->particle1 = getIndexOf(from_x, from_y);
        s->particle2 = getIndexOf(to_x, to_y);
        s->stiffness = stiffness;
        s->restLength = restLength;
    } else {
        return NULL;
    }
    return s;
}


std::vector<Vector3f> ClothSystem::evalF(std::vector<Vector3f> state)
{
    std::vector<Vector3f> f(state.size());
    
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
        if (i != 2*W-2) { //ignore top right corner
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
        }
        f[i] = v;
        f[i+1] = F;
        particleNum++;
    }
    
    return f;
}

//particle in the 1D state array at row i, column j (zero indexed)
int ClothSystem::getIndexOf(int i, int j)
{
    return (i * W) + j;
}

bool ClothSystem::isOnCloth(int i, int j)
{
    return i < W && j < H && i >= 0 && j >= 0;
}

void ClothSystem::draw(GLProgram& gl)
{
    //TODO 5: render the system 
    //         - ie draw the particles as little spheres
    //         - or draw the springs as little lines or cylinders
    //         - or draw wireframe mesh

    const Vector3f CLOTH_COLOR(0.9f, 0.9f, 0.9f);
    gl.updateMaterial(CLOTH_COLOR);

    for (int i = 0; i < ClothSystem::getState().size(); i+=2) {
        Vector3f pos = ClothSystem::getState()[i];
        gl.updateModelMatrix(Matrix4f::translation(pos));
        drawSphere(0.075f, 10, 10);
    }

    // EXAMPLE: This shows you how to render lines to debug the spring system.
    //
    //          You should replace this code.
    //
    //          Since lines don't have a clearly defined normal, we can't use
    //          a regular lighting model.
    //          GLprogram has a "color only" mode, where illumination
    //          is disabled, and you specify color directly as vertex attribute.
    //          Note: enableLighting/disableLighting invalidates uniforms,
    //          so you'll have to update the transformation/material parameters
    //          after a mode change.
//    gl.disableLighting();
//    gl.updateModelMatrix(Matrix4f::identity()); // update uniforms after mode change
//    VertexRecorder rec;
//    rec.record(O, CLOTH_COLOR);
//    rec.record(O + Vector3f(w, 0, 0), CLOTH_COLOR);
//    rec.record(O, CLOTH_COLOR);
//    rec.record(O + Vector3f(0, -w, 0), CLOTH_COLOR);
//
//    rec.record(O + Vector3f(w, 0, 0), CLOTH_COLOR);
//    rec.record(O + Vector3f(w, -w, 0), CLOTH_COLOR);
//
//    rec.record(O + Vector3f(0, -w, 0), CLOTH_COLOR);
//    rec.record(O + Vector3f(w, -w, 0), CLOTH_COLOR);
//    glLineWidth(3.0f);
//    rec.draw(GL_LINES);
//
//    gl.enableLighting(); // reset to default lighting model
    // EXAMPLE END
}

