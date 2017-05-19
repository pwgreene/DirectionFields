#include "gl.h"
#include <GLFW/glfw3.h>

#include <cmath>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <vector>
#include <sstream>
#include <cstdio>
#include <cstdint>

#include "vertexrecorder.h"
#include "recorder.h"
#include "starter3_util.h"
#include "camera.h"
#include "particlesystem.h"

using namespace std;

namespace
{
    
    // Declarations of functions whose implementations occur later.

    void initRendering();
    
    bool NPR_RENDER = true;

    // initialized in main()
    Vector3f MESH_COLOR;
    Vector3f MESH_AMBIENT;
    Vector3f FIELD_COLOR;
    Vector3f FIELD_AMBIENT;
    Vector3f BACKGROUND;
    
    // Some constants
    
    const Vector3f LIGHT_POS(0.0f, 0.0f, 22.0f);
    const Vector3f LIGHT_COLOR(500, 500, 500);
    
    const Vector3f SING_COLOR(0.f, 1.0f, 0.0f);
    const float ambient_amount = .15;

    
    // Globals here.

    // for rendering a mesh
    vector<Vector3f> vecv;
    vector<float> vecv_scale; //scaling factor per vertex
    vector<Vector3f> vecn;
    vector<vector<unsigned>> vecf;
    vector<Vector3f> X; //reference edge
    vector<vector<float>> field; //field angles
    vector<Vector3f> singularities;
    int hasSingularities;
    int fieldDegree;
    
    float FIELD_BAR_LENGTH;
    float SCALE;
    float FIELD_BAR_WIDTH;

    Camera camera;
    bool gMousePressed = false;
    GLuint program_color;
    GLuint program_light;

    // Function implementations
    static void keyCallback(GLFWwindow* window, int key,
        int scancode, int action, int mods)
    {
        if (action == GLFW_RELEASE) { // only handle PRESS and REPEAT
            return;
        }

        // Special keys (arrows, CTRL, ...) are documented
        // here: http://www.glfw.org/docs/latest/group__keys.html
        switch (key) {
        case GLFW_KEY_ESCAPE: // Escape key
            exit(0);
            break;
        case ' ':
        {
            Matrix4f eye = Matrix4f::identity();
            camera.SetRotation(eye);
            camera.SetCenter(Vector3f(0, 0, 0));
            break;
        }
        default:
            cout << "Unhandled key press " << key << "." << endl;
        }
    }

    static void mouseCallback(GLFWwindow* window, int button, int action, int mods)
    {
        double xd, yd;
        glfwGetCursorPos(window, &xd, &yd);
        int x = (int)xd;
        int y = (int)yd;

        int lstate = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT);
        int rstate = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT);
        int mstate = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_MIDDLE);
        if (lstate == GLFW_PRESS) {
            gMousePressed = true;
            camera.MouseClick(Camera::LEFT, x, y);
        }
        else if (rstate == GLFW_PRESS) {
            gMousePressed = true;
            camera.MouseClick(Camera::RIGHT, x, y);
        }
        else if (mstate == GLFW_PRESS) {
            gMousePressed = true;
            camera.MouseClick(Camera::MIDDLE, x, y);
        }
        else {
            gMousePressed = true;
            camera.MouseRelease(x, y);
            gMousePressed = false;
        }
    }

    static void motionCallback(GLFWwindow* window, double x, double y)
    {
        if (!gMousePressed) {
            return;
        }
        camera.MouseDrag((int)x, (int)y);
    }

    void setViewport(GLFWwindow* window)
    {
        int w, h;
        glfwGetFramebufferSize(window, &w, &h);

        camera.SetDimensions(w, h);
        camera.SetViewport(0, 0, w, h);
        camera.ApplyViewport();
    }

    
    void loadMesh()
    {
        //for .off files
        const int MAX_BUFFERSIZE = 4096;
        char buffer[MAX_BUFFERSIZE];
        cin.getline(buffer, MAX_BUFFERSIZE);
        cin.getline(buffer, MAX_BUFFERSIZE);
        stringstream ssn(buffer);
        float totalEdgeLen = 0;
        float totalEdgeForAvg = 0;
        
        vector<vector<int>> adjacentFaces; //keep track of adjacent faces to each vertex
        int nv, nt, ne;
        vector<Vector3f> faceNormals;
        ssn >> nv >> nt >> ne;
        printf("%d, %d, %d\n", nv, nt, ne);
        adjacentFaces.resize(nv);
        X.resize(nv);
        for (int v_i = 0; v_i < nv; v_i++) {
            Vector3f v;
            Vector3f n;
            cin.getline(buffer, MAX_BUFFERSIZE);
            stringstream ss(buffer);
            ss >> v[0] >> v[1] >> v[2];
            vecv.push_back(v);
        }
        printf("loaded %d vertices\n", nv);
        for (int f_i = 0; f_i < nt; f_i++) {
            vector<unsigned> f;
            unsigned x, a, b, c;
            cin.getline(buffer, MAX_BUFFERSIZE);
            stringstream ss(buffer);
            ss >> x >> a >> b >> c; //just assuming triangle mesh for simplicity
            f.push_back(a); f.push_back(a); f.push_back(a);
            f.push_back(b); f.push_back(b); f.push_back(b);
            f.push_back(c); f.push_back(c); f.push_back(c);
            if (a > nv || b > nv || c > nv) {
                printf("bad\n");
            }
            vecf.push_back(f);
            //compute face normals
            Vector3f ij = vecv[f[3]] - vecv[f[0]];
            Vector3f jk = vecv[f[6]] - vecv[f[3]];
            totalEdgeLen += ij.abs() + jk.abs();
            totalEdgeForAvg += 2;
//            X[f[0]] = ij.normalized(); X[f[3]] = -ij.normalized(); X[f[6]] = -jk.normalized(); //arbitrarily choose these--for now
            Vector3f n = Vector3f::cross(ij, jk);
            faceNormals.push_back(n.normalized());
            for (int i = 0; i < 9; i+=3) {
                adjacentFaces[f[i]].push_back(f_i);
            }
        }
        printf("loaded %d faces\n", nt);
        for (int v_i = 0; v_i < nv; v_i++) {
            Vector3f v_n = Vector3f(0, 0, 0);
            
            int nfaces = adjacentFaces[v_i].size();
            if (nfaces == 0) {
                printf("%d\n", v_i);
            }
            for (int i = 0; i < nfaces; i++) {
                v_n += faceNormals[adjacentFaces[v_i][i]];
            }
            v_n /= nfaces;
            v_n.normalize();
            
            vecn.push_back(v_n);
        }
        FIELD_BAR_LENGTH = totalEdgeLen / totalEdgeForAvg /2.0; // estimated avg edge length compuation - NOT EXACT!
        FIELD_BAR_WIDTH = FIELD_BAR_LENGTH * .02;
        SCALE = .5/FIELD_BAR_LENGTH;
        printf("%lu, %lu, %lu\n", vecv.size(), vecn.size(), vecf.size());
        printf("scale: %f\n", FIELD_BAR_LENGTH);
    }
    
    void loadField() {
        const int MAX_BUFFERSIZE = 4096;
        float v_scale;
        char buffer[MAX_BUFFERSIZE];
        cin.getline(buffer, MAX_BUFFERSIZE);
        stringstream ss(buffer);
        int nv, n;
        int ref_i, v_i;
        float theta; //assume just pass in angles rather than complex vectors
        ss >> nv >> n >> hasSingularities;
        field.resize(nv);
        for (int i = 0; i < nv; i++) {
            cin.getline(buffer, MAX_BUFFERSIZE);
            stringstream ss(buffer);
            ss >> v_i >> ref_i;
            for (int k = 0; k < n; k++) {
                ss >> theta;
                field[v_i].push_back(theta);
            }
            X[i] = (vecv[ref_i] - vecv[v_i]).normalized();
        }
        fieldDegree = n;
        printf("degree %d field loaded from file\n", fieldDegree);
        //load singularities
        if (hasSingularities) {
            int nSingularities, faceIndex;
            cin.getline(buffer, MAX_BUFFERSIZE);
            stringstream ss(buffer);
            ss >> nSingularities;
            for (int i = 0; i < nSingularities; i++) {
                cin.getline(buffer, MAX_BUFFERSIZE);
                ss = stringstream(buffer);
                ss >> faceIndex;
                Vector3f v1 = vecv[vecf[faceIndex][0]], v2 = vecv[vecf[faceIndex][3]], v3 = vecv[vecf[faceIndex][6]];
                Vector3f faceCenter = (v1 + v2 + v3) / 3.0;
                singularities.push_back(faceCenter);
            }
        }
        
    }
    
    void drawField() {
        GLProgram gl(program_light, program_color, &camera);
        gl.updateMaterial(FIELD_COLOR, FIELD_AMBIENT, Vector3f(0, 0, 0), 1.0, 1.0, BACKGROUND);
        for (int i = 0; i < vecv.size(); i++) {
            Vector3f a(0, 1, 0); //default orientation of cylinder
            Vector3f axis = Vector3f::cross(a, X[i]);
            for (int k = 0; k < fieldDegree; k++) {
                float phi = field[i][k];
                float angleToTangentPlane = acos(Vector3f::dot(a, X[i]));
                
                Matrix4f rotationToPlane = Matrix4f::rotation(axis, angleToTangentPlane);
                Matrix4f rotationInPlane = Matrix4f::rotation(vecn[i], phi);
                Matrix4f scaling = Matrix4f::uniformScaling(SCALE);
                Matrix4f translation = scaling*Matrix4f::translation(vecv[i]);//*Matrix4f::translation(Vector3f(0, -5, 0));
                gl.updateModelMatrix(translation*rotationInPlane*rotationToPlane);
                if (NPR_RENDER) {
//                    float scaleAmount = abs(1.0 - vecv_scale[i]);
//                    float scaling = .002/scaleAmount;
//                    if (scaling > 3 || abs(scaleAmount) <= 1e-8) {
//                        scaling = 3.;
//                    }
////                    printf("%d: %f", i, scaling);
                    float scaling = 1.6;
                    drawRect(FIELD_BAR_WIDTH*2, scaling*FIELD_BAR_LENGTH);
                } else
                    drawCylinder(3, FIELD_BAR_WIDTH*1.5, FIELD_BAR_LENGTH*.8);
            }
        }
        if (hasSingularities) {
            gl.updateMaterial(SING_COLOR, SING_COLOR);
            for (int i = 0; i < singularities.size(); i++) {
                Vector3f location = singularities[i];
                Matrix4f scaling = Matrix4f::uniformScaling(SCALE);
                Matrix4f translation = scaling*Matrix4f::translation(location);
                gl.updateModelMatrix(translation);
                drawSphere(FIELD_BAR_LENGTH*.5, 10, 10);
            }
        }
        
    }
    
    void drawMesh() {
        // draw obj mesh here
        // read vertices and face indices from vecv, vecn, vecf
        GLProgram gl(program_light, program_color, &camera);
        gl.updateLight(LIGHT_POS, LIGHT_COLOR.xyz()); // once per frame
        gl.updateMaterial(MESH_COLOR, MESH_AMBIENT, Vector3f(0, 0, 0), 1.0, 1.0, BACKGROUND);
        gl.updateModelMatrix(Matrix4f::uniformScaling(SCALE));
        GeometryRecorder rec(vecf.size() * 3);
        for (vector<unsigned> f : vecf) {
            rec.record(Vector3f(vecv[f[0]][0], vecv[f[0]][1], vecv[f[0]][2]),
                       Vector3f(vecn[f[2]][0], vecn[f[2]][1], vecn[f[2]][2]));
            
            rec.record(Vector3f(vecv[f[3]][0], vecv[f[3]][1], vecv[f[3]][2]),
                       Vector3f(vecn[f[5]][0], vecn[f[5]][1], vecn[f[5]][2]));
            
            rec.record(Vector3f(vecv[f[6]][0], vecv[f[6]][1], vecv[f[6]][2]),
                       Vector3f(vecn[f[8]][0], vecn[f[8]][1], vecn[f[8]][2]));
        }
        rec.draw();
        
    }
        
        

    //-------------------------------------------------------------------

    void initRendering()
    {
        // Clear to black
        glClearColor(0, 0, 0, 1);
        glEnable(GL_DEPTH_TEST);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    }
}

    // Main routine.
    // Set up OpenGL, define the callbacks and start the main loop
    int main(int argc, char** argv)
    {
        NPR_RENDER = stoi(argv[1]);
        if (NPR_RENDER) {
            cout << "Displaying with NPR texture\n";
            MESH_COLOR = Vector3f(1.0f, 1.0f, 1.0f);
            MESH_AMBIENT = 1.0 * MESH_COLOR;
            FIELD_COLOR = Vector3f(.6f, .6f, .6f);
            FIELD_AMBIENT = 0 * FIELD_COLOR;
            BACKGROUND = Vector3f(1.0f, 1.0f, 1.0f);
        } else {
            MESH_COLOR = Vector3f(0.1f, 0.1f, 0.8f);
            MESH_AMBIENT = .3 * MESH_COLOR;
            FIELD_COLOR = Vector3f(1.0f, 0.6f, 0.2f);
            FIELD_AMBIENT = .7 * FIELD_COLOR;
            BACKGROUND = Vector3f(0.0f, 0.0f, 0.0f);
        }
        
        GLFWwindow* window = createOpenGLWindow(1024, 1024, "Assignment 3");

        // setup the event handlers
        glfwSetKeyCallback(window, keyCallback);
        glfwSetMouseButtonCallback(window, mouseCallback);
        glfwSetCursorPosCallback(window, motionCallback);

        initRendering();

        // The program object controls the programmable parts
        // of OpenGL. All OpenGL programs define a vertex shader
        // and a fragment shader.
        program_color = compileProgram(c_vertexshader, c_fragmentshader_color);
        if (!program_color) {
            printf("Cannot compile program\n");
            return -1;
        }
        program_light = compileProgram(c_vertexshader, c_fragmentshader_light);
        if (!program_light) {
            printf("Cannot compile program\n");
            return -1;
        }

        camera.SetDimensions(600, 600);
        camera.SetPerspective(50);
        camera.SetDistance(20);

        loadMesh();
        loadField();
        
        // Main Loop
        while (!glfwWindowShouldClose(window)) {
            // Clear the rendering window
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

            setViewport(window);
            
            drawMesh();
            drawField();

            // Make back buffer visible
            glfwSwapBuffers(window);

            // Check if any input happened during the last frame
            glfwPollEvents();
        }

        // All OpenGL resource that are created with
        // glGen* or glCreate* must be freed.
        glDeleteProgram(program_color);
        glDeleteProgram(program_light);


        return 0;	// This line is never reached.
    }
