/*
    CSCI 420 Computer Graphics, USC
    Assignment 2: Roller Coaster
    C++ starter code
    Student username: ymurata
*/

#include "basicPipelineProgram.h"
#include "openGLMatrix.h"
#include "imageIO.h"
#include "openGLHeader.h"
#include "glutHeader.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <cstring>
#include <vector>
#include <cmath>

#if defined(WIN32) || defined(_WIN32)
    #ifdef _DEBUG
        #pragma comment(lib, "glew32d.lib")
    #else
        #pragma comment(lib, "glew32.lib")
    #endif
#endif

#if defined(WIN32) || defined(_WIN32)
    char shaderBasePath[1024] = SHADER_BASE_PATH;
#else
    char shaderBasePath[1024] = "../openGLHelper-starterCode";
#endif

using namespace std;

// represents one control point along the spline
// and also a vector
struct Point
{
    double x, y, z;

    // default constructor
    Point() { x = y = z = 0.0f; }

    // customized constructor
    Point(float x, float y, float z)
    {
        this->x = x;
        this->y = y;
        this->z = z;
    }

    // scale point or vector
    Point mult(float scale)
    {
        return Point(scale * this->x, scale * this->y, scale * this->z);
    }

    // addition of 2 points or vectors
    Point add(Point p)
    {
        return Point(this->x + p.x, this->y + p.y, this->z + p.z);
    }

    // get negation of point or vector
    Point neg() { return this->mult(-1.0f); }

    // get unit vector
    Point unit()
    {
        float magnitude = sqrt(this->x * this->x + this->y * this->y + this->z * this->z);
        return Point(this->mult(1.0f / magnitude));
    }

    // computes cross product
    Point crossProd(Point p)
    {
        return Point(this->y * p.z - this->z * p.y,
                     this->z * p.x - this->x * p.z,
                     this->x * p.y - this->y * p.x);
    }
};

// spline struct 
// contains how many control points the spline has, and an array of control points 
struct Spline 
{
    int numControlPoints;
    Point * points;
};

// the spline array 
Spline * splines;
// total number of splines 
int numSplines;

int mousePos[2]; // x,y coordinate of the mouse position
int leftMouseButton = 0; // 1 if pressed, 0 if not 
int middleMouseButton = 0; // 1 if pressed, 0 if not
int rightMouseButton = 0; // 1 if pressed, 0 if not

// interaction control state
typedef enum { ROTATE, TRANSLATE, SCALE } CONTROL_STATE;
CONTROL_STATE controlState = ROTATE;

// state of the world
float landRotate[3] = { 0.0f, 0.0f, 0.0f };
float landTranslate[3] = { 0.0f, 0.0f, 0.0f };
float landScale[3] = { 1.0f, 1.0f, 1.0f };

// window setting
int windowWidth = 1280;
int windowHeight = 720;
char windowTitle[512] = "CSCI 420 homework II";

// Shader
GLuint program, texProgram, texHandle;
OpenGLMatrix openGLMatrix; // open GL helper
// pipeline programs
BasicPipelineProgram pipelineProgram, texPipelineProgram;
// handles, modelview and rojectionmatrices
GLint h_modelViewMatrix, h_projectionMatrix,
      tex_h_modelViewMatrix, tex_h_projectionMatrix;

// container for positions, normals, texture positions and uv vectors
vector<float> normals, positions, pos, uvs;
// look at positions at each u value
vector<Point> eyePositions, fPoints, upVectors;

// vbos and vaos
GLuint vbo, vao, texVbo, texVao;

int trackIdx = 0;

// current tangent, normal and B vectors
Point currentT, currentN, currentB;

int frame = -1; // frame # for screenshots

int loadSplines(char * argv) 
{
    char * cName = (char *) malloc(128 * sizeof(char));
    FILE * fileList;
    FILE * fileSpline;
    int iType, i = 0, j, iLength;

    // load the track file 
    fileList = fopen(argv, "r");
    if (fileList == NULL) 
    {
        printf ("can't open file\n");
        exit(1);
    }
    
    // stores the number of splines in a global variable 
    fscanf(fileList, "%d", &numSplines);

    splines = (Spline*) malloc(numSplines * sizeof(Spline));

    // reads through the spline files 
    for (j = 0; j < numSplines; j++) 
    {
        i = 0;
        fscanf(fileList, "%s", cName);
        fileSpline = fopen(cName, "r");

        if (fileSpline == NULL) 
        {
            printf ("can't open file\n");
            exit(1);
        }

        // gets length for spline file
        fscanf(fileSpline, "%d %d", &iLength, &iType);

        // allocate memory for all the points
        splines[j].points = (Point *)malloc(iLength * sizeof(Point));
        splines[j].numControlPoints = iLength;

        // saves the data to the struct
        while (fscanf(fileSpline, "%lf %lf %lf", 
         &splines[j].points[i].x, 
         &splines[j].points[i].y, 
         &splines[j].points[i].z) != EOF) 
        {
            i++;
        }
    }

    free(cName);

    return 0;
}

int initTexture(const char * imageFilename, GLuint textureHandle)
{
    // read the texture image
    ImageIO img;
    ImageIO::fileFormatType imgFormat;
    ImageIO::errorType err = img.load(imageFilename, &imgFormat);

    if (err != ImageIO::OK) 
    {
        printf("Loading texture from %s failed.\n", imageFilename);
        return -1;
    }

    // check that the number of bytes is a multiple of 4
    if (img.getWidth() * img.getBytesPerPixel() % 4) 
    {
        printf("Error (%s): The width*numChannels in the loaded image must be a multiple of 4.\n", imageFilename);
        return -1;
    }

    // allocate space for an array of pixels
    int width = img.getWidth();
    int height = img.getHeight();
    unsigned char * pixelsRGBA = new unsigned char[4 * width * height]; // we will use 4 bytes per pixel, i.e., RGBA

    // fill the pixelsRGBA array with the image pixels
    memset(pixelsRGBA, 0, 4 * width * height); // set all bytes to 0
    for (int h = 0; h < height; h++)
        for (int w = 0; w < width; w++) 
        {
            // assign some default byte values (for the case where img.getBytesPerPixel() < 4)
            pixelsRGBA[4 * (h * width + w) + 0] = 0; // red
            pixelsRGBA[4 * (h * width + w) + 1] = 0; // green
            pixelsRGBA[4 * (h * width + w) + 2] = 0; // blue
            pixelsRGBA[4 * (h * width + w) + 3] = 255; // alpha channel; fully opaque

            // set the RGBA channels, based on the loaded image
            int numChannels = img.getBytesPerPixel();
            for (int c = 0; c < numChannels; c++) // only set as many channels as are available in the loaded image; the rest get the default value
                pixelsRGBA[4 * (h * width + w) + c] = img.getPixel(w, h, c);
        }

    // bind the texture
    glBindTexture(GL_TEXTURE_2D, textureHandle);

    // initialize the texture
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, pixelsRGBA);

    // generate the mipmaps for this texture
    glGenerateMipmap(GL_TEXTURE_2D);

    // set the texture parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    // query support for anisotropic texture filtering
    GLfloat fLargest;
    glGetFloatv(GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT, &fLargest);
    printf("Max available anisotropic samples: %f\n", fLargest);
    // set anisotropic texture filtering
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY_EXT, 0.5f * fLargest);

    // query for any errors
    GLenum errCode = glGetError();
    if (errCode != 0) 
    {
        printf("Texture initialization error. Error code: %d.\n", errCode);
        return -1;
    }
    
    // de-allocate the pixel array -- it is no longer needed
    delete [] pixelsRGBA;

    return 0;
}

/*
 * matrix calculation for spline positions and tangent at u
 * p(u) = [u^3 u^2 u 1] M C
 * t(u) = [3u^2 2u 1 0] M C
*/
Point matCalc(int mode, float u, float s, Point p1, Point p2, Point p3, Point p4)
{
    Point result;
    float temp[12];
    float basis[16] = { -s,       2.0f - s, s - 2.0f,        s,
                        2.0f * s, s - 3.0f, 3.0f - 2.0f * s, -s,
                        -s,       0.0f,     s,               0.0f,
                        0.0f,     1.0f,     0.0f,            0.0f };

    float vec4U[4];
    if (mode == 1) // Get a point with Catmull-Rom [u^3 u^2 u 1]
    {
		vec4U[0] = u * u * u; vec4U[1] = u * u; vec4U[2] = u; vec4U[3] = 1.0f;
    }
    else if (mode == 2) // Get a tangent [3u^2 2u 1 0]
    {
		vec4U[0] = 3 * u * u; vec4U[1] = 2 * u; vec4U[2] = 1.0f; vec4U[3] = 0.0f;
    }

    temp[0] = basis[0] * p1.x + basis[1] * p2.x + basis[2] * p3.x + basis[3] * p4.x;
    temp[1] = basis[0] * p1.y + basis[1] * p2.y + basis[2] * p3.y + basis[3] * p4.y;
    temp[2] = basis[0] * p1.z + basis[1] * p2.z + basis[2] * p3.z + basis[3] * p4.z;

    temp[3] = basis[4] * p1.x + basis[5] * p2.x + basis[6] * p3.x + basis[7] * p4.x;
    temp[4] = basis[4] * p1.y + basis[5] * p2.y + basis[6] * p3.y + basis[7] * p4.y;
    temp[5] = basis[4] * p1.z + basis[5] * p2.z + basis[6] * p3.z + basis[7] * p4.z;

    temp[6] = basis[8] * p1.x + basis[9] * p2.x + basis[10] * p3.x + basis[11] * p4.x;
    temp[7] = basis[8] * p1.y + basis[9] * p2.y + basis[10] * p3.y + basis[11] * p4.y;
    temp[8] = basis[8] * p1.z + basis[9] * p2.z + basis[10] * p3.z + basis[11] * p4.z;

    temp[9] = basis[12] * p1.x + basis[13] * p2.x + basis[14] * p3.x + basis[15] * p4.x;
    temp[10] = basis[12] * p1.y + basis[13] * p2.y + basis[14] * p3.y + basis[15] * p4.y;
    temp[11] = basis[12] * p1.z + basis[13] * p2.z + basis[14] * p3.z + basis[15] * p4.z;

    result.x = vec4U[0] * temp[0] + vec4U[1] * temp[3] + vec4U[2] * temp[6] + vec4U[3] * temp[9];
    result.y = vec4U[0] * temp[1] + vec4U[1] * temp[4] + vec4U[2] * temp[7] + vec4U[3] * temp[10];
    result.z = vec4U[0] * temp[2] + vec4U[1] * temp[5] + vec4U[2] * temp[8] + vec4U[3] * temp[11];

    return result;
}

// write a screenshot to the specified filename
void saveScreenshot(const char * filename)
{
    unsigned char * screenshotData = new unsigned char[windowWidth * windowHeight * 3];
    glReadPixels(0, 0, windowWidth, windowHeight, GL_RGB, GL_UNSIGNED_BYTE, screenshotData);

    ImageIO screenshotImg(windowWidth, windowHeight, 3, screenshotData);

    if (screenshotImg.save(filename, ImageIO::FORMAT_JPEG) == ImageIO::OK)
        cout << "File " << filename << " saved successfully." << endl;
    else cout << "Failed to save file " << filename << '.' << endl;

    delete [] screenshotData;
}

void setTextureUnit(GLint unit)
{
    // select the active texture unit
    glActiveTexture(unit);
    //get a handle to the "textureImage" shader variable
    GLint h_textureImage = glGetUniformLocation(texProgram, "textureImage");
    // deem the shader variable "textureImage" to read from texture unit "unit"
    glUniform1i(h_textureImage, unit - GL_TEXTURE0);
}

/*
 * Camera setting
 */
void setView()
{
    //openGLMatrix.LookAt(0, 0, -10, 0, 0, 0, 0, 1, 0); //milestone hw2
    openGLMatrix.LookAt(eyePositions.at(trackIdx).x, eyePositions.at(trackIdx).y, eyePositions.at(trackIdx).z,
                        fPoints.at(trackIdx).x, fPoints.at(trackIdx).y, fPoints.at(trackIdx).z,
                        upVectors.at(trackIdx).x, upVectors.at(trackIdx).y, upVectors.at(trackIdx).z);
}

/*
 * Set transformation
 */
void setTransform()
{
    // Transformation
    openGLMatrix.Translate(landTranslate[0], landTranslate[1], landTranslate[2]);
    openGLMatrix.Rotate(landRotate[0], 1.0, 0.0, 0.0);
    openGLMatrix.Rotate(landRotate[1], 0.0, 1.0, 0.0);
    openGLMatrix.Rotate(landRotate[2], 0.0, 0.0, 1.0);
    openGLMatrix.Scale(landScale[0], landScale[1], landScale[2]); // scale the onject
}

/*
 * Set model-view and projection matrix
 */
void setModelViewProjectionMatrix()
{
    // ModelView Matrix
    // prepare the modelview matrix
    openGLMatrix.SetMatrixMode(OpenGLMatrix::ModelView);
    openGLMatrix.LoadIdentity();

    setView();
    setTransform();

    float m[16]; // column-major
    openGLMatrix.GetMatrix(m);
    // ipload m to shader
    glUniformMatrix4fv(tex_h_modelViewMatrix, 1, GL_FALSE, m);

    // Projection Matrix
    openGLMatrix.SetMatrixMode(OpenGLMatrix::Projection);
    float p[16]; // column-major
    openGLMatrix.GetMatrix(p);
    // upload p to shader
    glUniformMatrix4fv(tex_h_projectionMatrix, 1, GL_FALSE, p);
}

float view[16]; // container of view matrix

/*
 * Calculate view light direction
 * Note: lightDirection is a column major
 */
Point getView(Point lightDirection)
{
    Point result;
    result.x = view[0] * lightDirection.x + view[4] * lightDirection.y + view[8] * lightDirection.z;
    result.y = view[1] * lightDirection.x + view[5] * lightDirection.y + view[9] * lightDirection.z;
    result.z = view[2] * lightDirection.x + view[6] * lightDirection.y + view[10] * lightDirection.z;
    return result;
}

/*
 * Set phong shading
 */
void setPhongShading()
{
    // Set model-view matrix
    openGLMatrix.SetMatrixMode(OpenGLMatrix::ModelView);
    openGLMatrix.LoadIdentity();
    setView();

    //float view[16];
    openGLMatrix.GetMatrix(view); // read the view matrix

    Point lightDirection = Point(0, 1, 0); // the "sun" at noon (this is a vector)
    GLint h_viewLightDirection = glGetUniformLocation(program, "viewLightDirection");
    Point lD = getView(lightDirection);
    // light direction in the view space
    float viewLightDirection[3] = { static_cast<float>(lD.x), static_cast<float>(lD.y), static_cast<float>(lD.z) };
    //viewLightDirection = (view * float4(lightDirection, 0.0)).xyz; // <---------------------------------------
    // upload viewLightDirection to the GPU
    glUniform3fv(h_viewLightDirection, 1, viewLightDirection);

    /////////////////////////////////////////////////////////////////////////////////////////

    float intensity = 0.7f;
    float materialCoeff = 0.9f;

    GLint h_La = glGetUniformLocation(program, "La");
    float La[4] = { intensity, intensity, intensity, 1.0f };
    glUniform4fv(h_La, 1, La);

    GLint h_ka = glGetUniformLocation(program, "ka");
    float ka[4] = { materialCoeff, materialCoeff, materialCoeff, materialCoeff };
    glUniform4fv(h_ka, 1, ka);

    GLint h_Ld = glGetUniformLocation(program, "Ld");
    float Ld[4] = { intensity, intensity, intensity, 1.0f };
    glUniform4fv(h_Ld, 1, Ld);

    GLint h_kd = glGetUniformLocation(program, "kd");
    float kd[4] = { materialCoeff, materialCoeff, materialCoeff, materialCoeff };
    glUniform4fv(h_kd, 1, kd);

    GLint h_Ls = glGetUniformLocation(program, "Ls");
    float Ls[4] = { intensity, intensity, intensity, intensity };
    glUniform4fv(h_Ls, 1, Ls);

    GLint h_ks = glGetUniformLocation(program, "ks");
    float ks[4] = { materialCoeff, materialCoeff, materialCoeff, materialCoeff };
    glUniform4fv(h_ks, 1, ks);

    GLint h_alpha = glGetUniformLocation(program, "alpha");
    glUniform1f(h_alpha, 1.0f);

    /////////////////////////////////////////////////////////////////////////////////////////

    // Transformation
    setTransform();

    float m[16]; // column-major
    openGLMatrix.GetMatrix(m);
    glUniformMatrix4fv(h_modelViewMatrix, 1, GL_FALSE, m);

    // Projection Matrix
    openGLMatrix.SetMatrixMode(OpenGLMatrix::Projection);
    float p[16]; // column-major
    openGLMatrix.GetMatrix(p);
    glUniformMatrix4fv(h_projectionMatrix, 1, GL_FALSE, p);

    GLint h_normalMatrix = glGetUniformLocation(program, "normalMatrix");
    float n[16];
    openGLMatrix.SetMatrixMode(OpenGLMatrix::ModelView);
    openGLMatrix.GetNormalMatrix(n); // get normal matrix

    // upload n to the GPU
    GLboolean isRowMajor = GL_FALSE;
    glUniformMatrix4fv(h_normalMatrix, 1, isRowMajor, n);
}

void displayFunc()
{
    // clear the color and depth buffers
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    GLint first = 0;
    
    // Render spline /////////////////////////////////////////////////////////

    // Bind basic pipeline program
    pipelineProgram.Bind();
    setPhongShading();
    glBindVertexArray(vao); // bind the VAO
    glDrawArrays(GL_TRIANGLES, first, positions.size());

    // Render texture map ////////////////////////////////////////////////////

    // select the active texture unit
    setTextureUnit(GL_TEXTURE0);
    // select the texture to use ("texHandle" was generated by glGenTextures)
    glBindTexture(GL_TEXTURE_2D, texHandle);

    // Bind texture pipeline program
    texPipelineProgram.Bind();
    setModelViewProjectionMatrix();
    glBindVertexArray(texVao); // bind the VAO
    glDrawArrays(GL_TRIANGLES, first, pos.size());

    //////////////////////////////////////////////////////////////////////////

    glBindVertexArray(0); // unbind the VAO
    glutSwapBuffers(); // swap the buffers:
}

// Set current tangent, normal and B vectors at u
void setTNB(float u, int sCtrlPoint)
{
    // 4 control points
    Point p1, p2, p3, p4;

    try
    {
        // Get 4 control points
        p1 = splines[0].points[sCtrlPoint];
        p2 = splines[0].points[sCtrlPoint+1];
        p3 = splines[0].points[sCtrlPoint+2];
        p4 = splines[0].points[sCtrlPoint+3];

        // Set tangent
        currentT = matCalc(2, u, 0.5f, p1, p2, p3, p4).unit();

        // Set normal
        if (u == 0.00f && sCtrlPoint == 0)
        {
            currentN = currentT.crossProd(Point(0.0f, -1.0f, 0.0f)).unit();
        }
        else
        {
            currentN = currentB.crossProd(currentT).unit(); // this currentB is previous B
        }

        // Set B vector
        currentB = currentT.crossProd(currentN).unit();
    }
    catch (out_of_range& e) {}
}

bool skip = true;
void idleFunc()
{
    // make the screen update 
    glutPostRedisplay();

    // Update eye position
    trackIdx++;
    if (trackIdx == eyePositions.size()) trackIdx = 0;

    // take 1000 screenshots
    /*
    if (frame == -1)
    {
        frame++;
        return;
    }
    if (skip)
    {
        skip = false;
        return;
    }
    if (frame < 1000)
    {
        saveScreenshot(("screenshots/screenshot" + string(3 - to_string(frame).length(), '0') + to_string(frame) + ".jpg").c_str());
        frame++;
        skip = true;
    } else
    {
        cout << "End of video!!!!!!!!!" << endl;
    }
    */
}

void reshapeFunc(int w, int h)
{
    glViewport(0, 0, w, h);

    // setup perspective matrix
    openGLMatrix.SetMatrixMode(OpenGLMatrix::Projection);
    openGLMatrix.LoadIdentity();
    //openGLMatrix.Perspective(45.0, 1.0 * w / h, 0.01, 5000.0);
    openGLMatrix.Perspective(45.0, 1.0 * w / h, 0.01, 100.0);
    openGLMatrix.SetMatrixMode(OpenGLMatrix::ModelView);
}

void mouseMotionDragFunc(int x, int y)
{
    // mouse has moved and one of the mouse buttons is pressed (dragging)

    // the change in mouse position since the last invocation of this function
    int mousePosDelta[2] = { x - mousePos[0], y - mousePos[1] };

    switch (controlState)
    {
        // translate the landscape
        case TRANSLATE:
            if (leftMouseButton)
            {
                // control x,y translation via the left mouse button
                landTranslate[0] += mousePosDelta[0] * 0.01f;
                landTranslate[1] -= mousePosDelta[1] * 0.01f;
            }
            if (middleMouseButton)
            {
                // control z translation via the middle mouse button
                landTranslate[2] += mousePosDelta[1] * 0.01f;
            }
            break;

        // rotate the landscape
        case ROTATE:
            if (leftMouseButton)
            {
                // control x,y rotation via the left mouse button
                landRotate[0] += mousePosDelta[1];
                landRotate[1] += mousePosDelta[0];
            }
            if (middleMouseButton)
            {
                // control z rotation via the middle mouse button
                landRotate[2] += mousePosDelta[1];
            }
            break;

        // scale the landscape
        case SCALE:
            if (leftMouseButton)
            {
                // control x,y scaling via the left mouse button
                landScale[0] *= 1.0f + mousePosDelta[0] * 0.01f;
                landScale[1] *= 1.0f - mousePosDelta[1] * 0.01f;
            }
            if (middleMouseButton)
            {
                // control z scaling via the middle mouse button
                landScale[2] *= 1.0f - mousePosDelta[1] * 0.01f;
            }
            break;
    }

    // store the new mouse position
    mousePos[0] = x;
    mousePos[1] = y;
}

void mouseMotionFunc(int x, int y)
{
    // mouse has moved
    // store the new mouse position
    mousePos[0] = x;
    mousePos[1] = y;
}

void mouseButtonFunc(int button, int state, int x, int y)
{
    // a mouse button has has been pressed or depressed

    // keep track of the mouse button state, in leftMouseButton, middleMouseButton, rightMouseButton variables
    switch (button)
    {
        case GLUT_LEFT_BUTTON:
            leftMouseButton = (state == GLUT_DOWN);
        break;

        case GLUT_MIDDLE_BUTTON:
            middleMouseButton = (state == GLUT_DOWN);
        break;

        case GLUT_RIGHT_BUTTON:
            rightMouseButton = (state == GLUT_DOWN);
        break;
    }

    // keep track of whether CTRL and SHIFT keys are pressed
    switch (glutGetModifiers())
    {
        case GLUT_ACTIVE_SHIFT:
            controlState = SCALE;
        break;

        // if CTRL and SHIFT are not pressed, we are in rotate mode
        default:
            controlState = ROTATE;
        break;
    }

    // store the new mouse position
    mousePos[0] = x;
    mousePos[1] = y;
}

void keyboardFunc(unsigned char key, int x, int y)
{
    int mousePosDelta[2] = { x - mousePos[0], y - mousePos[1] };

    switch (key)
    {
        case 27: // ESC key
            exit(0); // exit the program
        break;

        case 'x':
            // take a screenshot
            saveScreenshot(("screenshots/screenshot" + string(3 - to_string(frame).length(), '0') + to_string(frame) + ".jpg").c_str());
            frame++;
        break;

        // use t key to translate object
        case 't':
            controlState = TRANSLATE;
        break;
    }

    // store the new mouse position
    mousePos[0] = x;
    mousePos[1] = y;
}

// initialize VBO
void initVBO(int mode)
{
    if (mode == 1)
    {
        // init vbo’s size, but don’t assign any data to it
        glBufferData(GL_ARRAY_BUFFER, sizeof(float) * (positions.size() + normals.size()), nullptr, GL_STATIC_DRAW);
        // upload position data
        glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(float) * positions.size(), static_cast<void*>(positions.data()));
        // upload color data
        glBufferSubData(GL_ARRAY_BUFFER, sizeof(float) * positions.size(), sizeof(float) * normals.size(), static_cast<void*>(normals.data()));
    }
    else
    {
        // init vbo's size, but don’t assign any data to it
        glBufferData(GL_ARRAY_BUFFER, sizeof(float) * (pos.size() + uvs.size()), nullptr, GL_STATIC_DRAW);
        // upload pos data
        glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(float) * pos.size(), static_cast<void*>(pos.data()));
        // upload uvs data
        glBufferSubData(GL_ARRAY_BUFFER, sizeof(float) * pos.size(), sizeof(float) * uvs.size(), static_cast<void*>(uvs.data()));
    }
}

/*
 * Add positions and normals of a point
 */
void  addPoint(Point p, Point normal)
{
    positions.push_back(p.x);
    positions.push_back(p.y);
    positions.push_back(p.z);

    normals.push_back(normal.x);
    normals.push_back(normal.y);
    normals.push_back(normal.z);
}

/*
 * Add points consisting of a triangle
 */
void addPoints(Point p1, Point p2, Point p3, Point normal)
{
    addPoint(p1, normal);
    addPoint(p2, normal);
    addPoint(p3, normal);
}

/*
 * Get a vertex from a central point with vectors
 */
Point getPoint(Point p, float scale, Point n, Point b)
{
    return p.add(n.add(b).mult(scale));
}

/*
 * Set camera positions (add positions to stack)
 */
void setCameraPosition(Point p)
{
    eyePositions.push_back(p.add(currentN));
    fPoints.push_back(p.add(currentN).add(currentT));
    upVectors.push_back(currentN);
}

/*
 * Add spline segments as triangles with respect to u
 */
void addTriangles(float u, int sCtrlPoint)
{
    Point p0, p1;
    Point v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15;
    Point v0_2, v1_2, v2_2, v3_2, v4_2, v5_2, v6_2, v7_2, v8_2, v9_2, v10_2, v11_2, v12_2, v13_2, v14_2, v15_2;
    float scale = 0.1f;
    float interval = 1.0f;//0.3f;

    // Get starting point
    p0 = matCalc(1, u, 0.5f, splines[0].points[sCtrlPoint], splines[0].points[sCtrlPoint+1], splines[0].points[sCtrlPoint+2], splines[0].points[sCtrlPoint+3]);
    setTNB(u, sCtrlPoint);
    setCameraPosition(p0.add(currentB.mult(0.5f * interval)));

    v0 = getPoint(p0, scale, Point(), currentB);
    v1 = getPoint(p0, scale, currentN, currentB);
    v2 = getPoint(p0, scale, currentN, currentB.neg());
    v3 = getPoint(p0, scale, Point(), currentB.neg());

    v8 = getPoint(p0, scale, currentN.neg(), currentB.mult(0.5f));
    v9 = getPoint(p0, scale, Point(), currentB.mult(0.5f));
    v10 = getPoint(p0, scale, Point(), currentB.mult(0.5f).neg());
    v11 = getPoint(p0, scale, currentN.neg(), currentB.mult(0.5f).neg());

    v0_2 = getPoint(p0.add(currentB.mult(interval)), scale, Point(), currentB);
    v1_2 = getPoint(p0.add(currentB.mult(interval)), scale, currentN, currentB);
    v2_2 = getPoint(p0.add(currentB.mult(interval)), scale, currentN, currentB.neg());
    v3_2 = getPoint(p0.add(currentB.mult(interval)), scale, Point(), currentB.neg());

    v8_2 = getPoint(p0.add(currentB.mult(interval)), scale, currentN.neg(), currentB.mult(0.5f));
    v9_2 = getPoint(p0.add(currentB.mult(interval)), scale, Point(), currentB.mult(0.5f));
    v10_2 = getPoint(p0.add(currentB.mult(interval)), scale, Point(), currentB.mult(0.5f).neg());
    v11_2 = getPoint(p0.add(currentB.mult(interval)), scale, currentN.neg(), currentB.mult(0.5f).neg());

    // Get ending point
    p1 = matCalc(1, u+0.01f, 0.5f, splines[0].points[sCtrlPoint], splines[0].points[sCtrlPoint+1], splines[0].points[sCtrlPoint+2], splines[0].points[sCtrlPoint+3]);
    setTNB(u+0.01f, sCtrlPoint);

    v4 = getPoint(p1, scale, Point(), currentB);
    v5 = getPoint(p1, scale, currentN, currentB);
    v6 = getPoint(p1, scale, currentN, currentB.neg());
    v7 = getPoint(p1, scale, Point(), currentB.neg());

    v12 = getPoint(p1, scale, currentN.neg(), currentB.mult(0.5f));
    v13 = getPoint(p1, scale, Point(), currentB.mult(0.5f));
    v14 = getPoint(p1, scale, Point(), currentB.mult(0.5f).neg());
    v15 = getPoint(p1, scale, currentN.neg(), currentB.mult(0.5f).neg());

    v4_2 = getPoint(p1.add(currentB.mult(interval)), scale, Point(), currentB);
    v5_2 = getPoint(p1.add(currentB.mult(interval)), scale, currentN, currentB);
    v6_2 = getPoint(p1.add(currentB.mult(interval)), scale, currentN, currentB.neg());
    v7_2 = getPoint(p1.add(currentB.mult(interval)), scale, Point(), currentB.neg());

    v12_2 = getPoint(p1.add(currentB.mult(interval)), scale, currentN.neg(), currentB.mult(0.5f));
    v13_2 = getPoint(p1.add(currentB.mult(interval)), scale, Point(), currentB.mult(0.5f));
    v14_2 = getPoint(p1.add(currentB.mult(interval)), scale, Point(), currentB.mult(0.5f).neg());
    v15_2 = getPoint(p1.add(currentB.mult(interval)), scale, currentN.neg(), currentB.mult(0.5f).neg());

    // rail 1
    // upper half
    addPoints(v0, v1, v3, currentT.neg()); addPoints(v2, v1, v3, currentT.neg());
    addPoints(v1, v0, v5, currentB);       addPoints(v4, v0, v5, currentB);
    addPoints(v2, v1, v6, currentN);       addPoints(v5, v1, v6, currentN);
    addPoints(v3, v2, v7, currentB.neg()); addPoints(v6, v2, v7, currentB.neg());
    addPoints(v3, v0, v7, currentN.neg()); addPoints(v4, v0, v7, currentN.neg());
    addPoints(v5, v4, v6, currentT);       addPoints(v7, v4, v6, currentT);

    // bottom half
    addPoints(v8, v9, v11, currentT.neg());   addPoints(v10, v9, v11, currentT.neg());
    addPoints(v9, v8, v13, currentB);         addPoints(v12, v8, v13, currentB);
    addPoints(v10, v9, v14, currentN);        addPoints(v13, v9, v14, currentN);
    addPoints(v11, v10, v15, currentB.neg()); addPoints(v14, v10, v15, currentB.neg());
    addPoints(v11, v8, v15, currentN.neg());  addPoints(v12, v8, v15, currentN.neg());
    addPoints(v13, v12, v14, currentT);       addPoints(v15, v12, v14, currentT);

    // rail 2
    // upper half
    addPoints(v0_2, v1_2, v3_2, currentT.neg());  addPoints(v2_2, v1_2, v3_2, currentT.neg());
    addPoints(v1_2, v0_2, v5_2, currentB);        addPoints(v4_2, v0_2, v5_2, currentB);
    addPoints(v2_2, v1_2, v6_2, currentN);        addPoints(v5_2, v1_2, v6_2, currentN);
    addPoints(v3_2, v2_2, v7_2, currentB.neg());  addPoints(v6_2, v2_2, v7_2, currentB.neg());
    addPoints(v3_2, v0_2, v7_2, currentN.neg());  addPoints(v4_2, v0_2, v7_2, currentN.neg());
    addPoints(v5_2, v4_2, v6_2, currentT);        addPoints(v7_2, v4_2, v6_2, currentT);

    // bottom half
    addPoints(v8_2, v9_2, v11_2, currentT.neg());   addPoints(v10_2, v9_2, v11_2, currentT.neg());
    addPoints(v9_2, v8_2, v13_2, currentB);         addPoints(v12_2, v8_2, v13_2, currentB);
    addPoints(v10_2, v9_2, v14_2, currentN);        addPoints(v13_2, v9_2, v14_2, currentN);
    addPoints(v11_2, v10_2, v15_2, currentB.neg()); addPoints(v14_2, v10_2, v15_2, currentB.neg());
    addPoints(v11_2, v8_2, v15_2, currentN.neg());  addPoints(v12_2, v8_2, v15_2, currentN.neg());
    addPoints(v13_2, v12_2, v14_2, currentT);       addPoints(v15_2, v12_2, v14_2, currentT);
}

// read spline data and fill positions and normals
void getData()
{
    // Create geometry
    //for (i = 0; i < numSplines; i++)
    //{
    for (int sCtrlPoint = 0; sCtrlPoint < splines[0].numControlPoints - 3; sCtrlPoint++)
    {
        for (float u = 0.0f; u < 1.0f; u += 0.01f)
            addTriangles(u, sCtrlPoint);
    }
    //}

    // Ground plane
    float size = 100.0f;
    float height = 20.0f;

    // ground positions
    pos.push_back(size); pos.push_back(-size); pos.push_back(height);
    pos.push_back(size); pos.push_back(size); pos.push_back(height);
    pos.push_back(-size); pos.push_back(size); pos.push_back(height);

    pos.push_back(-size); pos.push_back(size); pos.push_back(height);
    pos.push_back(-size); pos.push_back(-size); pos.push_back(height);
    pos.push_back(size); pos.push_back(-size); pos.push_back(height);

    // UVs
    float scale = 40.0f;
    uvs.push_back(scale); uvs.push_back(scale);
    uvs.push_back(scale); uvs.push_back(0.0f);
    uvs.push_back(0.0f); uvs.push_back(0.0f);

    uvs.push_back(0.0f); uvs.push_back(0.0f);
    uvs.push_back(0.0f); uvs.push_back(scale);
    uvs.push_back(scale); uvs.push_back(scale);
}

void loadSplineData(int argc, char *argv[])
{
    // load the splines from the provided filename
    loadSplines(argv[1]);

    printf("Loaded %d spline(s).\n", numSplines);
    for(int i=0; i<numSplines; i++)
        printf("Num control points in spline %d: %d.\n", i, splines[i].numControlPoints);
}

void initScene(int argc, char *argv[])
{
    glClearColor(0.56f, 0.871f, 1.0f, 0.0f);

    // enable hidden surface removal:
    glEnable(GL_DEPTH_TEST);

    loadSplineData(argc, argv);

    // read file and fill positions and normals
    getData();

    // Basic Shader /////////////////////////////////////////////////////////////////////

    bool isBasicShader = true;
    pipelineProgram.Init("../openGLHelper-starterCode", isBasicShader);
    // bind the pipeline program (run this before glUniformMatrix4fv)
    pipelineProgram.Bind(); // need this to render spline

    // Create handles for modelview and projection matrices
    program = pipelineProgram.GetProgramHandle();
    h_modelViewMatrix = glGetUniformLocation(program, "modelViewMatrix");
    h_projectionMatrix = glGetUniformLocation(program, "projectionMatrix");

    // set up vbo and vao for Splines

    // create VBO
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);

    // create VAO
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao); // bind the VAO

    initVBO(1); // create VBO

    GLboolean normalized = GL_FALSE;
    GLsizei stride = 0;

    // bind the VBO "buffer" (must be previously created)
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    // get location index of the "position" shader variable
    GLuint loc = glGetAttribLocation(program, "position");
    glEnableVertexAttribArray(loc); // enable the "position" attribute
    const void * offset = (const void*) 0;
    // set the layout of the "position" attribute data
    glVertexAttribPointer(loc, 3, GL_FLOAT, normalized, stride, offset);

    // get the location index of the "color" shader variable
    loc = glGetAttribLocation(program, "normal");
    glEnableVertexAttribArray(loc); // enable the "color" attribute
    offset = (const void*) (sizeof(float) * positions.size());
    // set the layout of the "color" attribute data
    glVertexAttribPointer(loc, 3, GL_FLOAT, normalized, stride, offset);

    // Texture Mapping Shader //////////////////////////////////////////////////////////////////////////

    isBasicShader = false;
    texPipelineProgram.Init("../openGLHelper-starterCode", isBasicShader);
    texPipelineProgram.Bind(); // need to disable to render spline (why????)
    
    texProgram = texPipelineProgram.GetProgramHandle();
    tex_h_modelViewMatrix = glGetUniformLocation(texProgram, "modelViewMatrix");
    tex_h_projectionMatrix = glGetUniformLocation(texProgram, "projectionMatrix");

    // create an integer handle for the texture
    glGenTextures(1, &texHandle);

    int code = initTexture("./ground_img/lawn.jpg", texHandle);
    if (code != 0)
    {
        printf("Error loading the texture image.\n");
        exit(EXIT_FAILURE);
    }

    // create VBO
    glGenBuffers(1, &texVbo);
    glBindBuffer(GL_ARRAY_BUFFER, texVbo);

    // create VAO
    glGenVertexArrays(1, &texVao);
    glBindVertexArray(texVao); // bind the VAO

    // read file and fill pos and uvs
    initVBO(2); // create VBO    

    glBindBuffer(GL_ARRAY_BUFFER, texVbo);

    // get location index of the "pos" shader variable
    loc = glGetAttribLocation(texProgram, "pos");
    glEnableVertexAttribArray(loc); // enable the "pos" attribute
    // set the layout of the "pos" attribute data
    offset = (const void*) 0;
    glVertexAttribPointer(loc, 3, GL_FLOAT, normalized, stride, offset);

    // get location index of the "texCoord" shader variable
    loc = glGetAttribLocation(texProgram, "texCoord");
    glEnableVertexAttribArray(loc); // enable the "texCoord" attribute
    // set the layout of the "texCoord" attribute data
    offset = (const void*) (sizeof(float) * pos.size());
    glVertexAttribPointer(loc, 2, GL_FLOAT, normalized, stride, offset);

    glBindVertexArray(0); // unbind the VAO
}

int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        printf ("usage: %s <trackfile>\n", argv[0]);
        exit(0);
    }

    cout << "Initializing GLUT..." << endl;
    glutInit(&argc,argv);

    cout << "Initializing OpenGL..." << endl;

    #ifdef __APPLE__
        glutInitDisplayMode(GLUT_3_2_CORE_PROFILE | GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH | GLUT_STENCIL);
    #else
        glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH | GLUT_STENCIL);
    #endif

    glutInitWindowSize(windowWidth, windowHeight);
    glutInitWindowPosition(0, 0);    
    glutCreateWindow(windowTitle);

    cout << "OpenGL Version: " << glGetString(GL_VERSION) << endl;
    cout << "OpenGL Renderer: " << glGetString(GL_RENDERER) << endl;
    cout << "Shading Language Version: " << glGetString(GL_SHADING_LANGUAGE_VERSION) << endl;

    // tells glut to use a particular display function to redraw 
    glutDisplayFunc(displayFunc);
    // perform animation inside idleFunc
    glutIdleFunc(idleFunc);
    // callback for mouse drags
    glutMotionFunc(mouseMotionDragFunc);
    // callback for idle mouse movement
    glutPassiveMotionFunc(mouseMotionFunc);
    // callback for mouse button changes
    glutMouseFunc(mouseButtonFunc);
    // callback for resizing the window
    glutReshapeFunc(reshapeFunc);
    // callback for pressing the keys on the keyboard
    glutKeyboardFunc(keyboardFunc);

    // init glew
    #ifdef __APPLE__
        // nothing is needed on Apple
    #else
        // Windows, Linux
        GLint result = glewInit();
        if (result != GLEW_OK)
        {
            cout << "error: " << glewGetErrorString(result) << endl;
            exit(EXIT_FAILURE);
        }
    #endif

    // do initialization
    initScene(argc, argv);

    // sink forever into the glut loop
    glutMainLoop();
}

