#pragma once
#include <glad/glad.h>
#include <glm/glm.hpp>
#include <GLFW/glfw3.h>
#include "lib_link.h"
#include <opencv2/opencv.hpp>

#include <glm/gtc/matrix_transform.hpp>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/ext.hpp>
#include <glm/gtx/string_cast.hpp>

#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>


	// Defines several possible options for camera movement. Used as abstraction to stay away from window-system specific input methods
enum Camera_Movement {
	FORWARD,
	BACKWARD,
	LEFT,
	RIGHT
};

// Default camera values
const float YAW = -90.0f;
const float PITCH = 0.0f;
const float SPEED = 2.5f;
const float SENSITIVITY = 0.1f;
const float ZOOM = 45.0f;

//*******************
// Camera Class
//*******************
class Camera {
public:
	// Camera Attributes
	glm::vec3 Position;
	glm::vec3 Front;
	glm::vec3 Up;
	glm::vec3 Right;
	glm::vec3 WorldUp;
	// Euler Angles
	float Yaw;
	float Pitch;
	// Camera options
	float MovementSpeed;
	float MouseSensitivity;
	float Zoom;
	float orthoZoom;

	// Constructor with vectors
	Camera(glm::vec3 position = glm::vec3(0.0f, 0.0f, 0.0f),
		glm::vec3 up = glm::vec3(0.0f, 1.0f, 0.0f),
		float yaw = YAW, float pitch = PITCH);
	Camera(float posX, float posY, float posZ, float upX, float upY, float upZ, float yaw, float pitch);

	Camera() {};
	~Camera() {};

	void poseCamera(float posX, float posY, float posZ, float upX, float upY, float upZ, float yaw, float pitch);
	glm::vec3 startingPosition;
	void resetCameraPosition();

	glm::mat4 GetViewMatrix();
	void ProcessKeyboard(Camera_Movement direction, float deltaTime);
	void ProcessMouseMovement(float xoffset, float yoffset, GLboolean constrainPitch = true);
	void ProcessMouseScroll(float yoffset);
	void ProcessMouseScrollOrthographic(float yoffset);
	void updateCameraVectors();
};

//***********************
// Shader class
//***********************
class Shader {
public:
	unsigned int ID;
	Shader(const char* vertexPath, const char* fragmentPath, const char* geometryPath = nullptr);
	void use();
	void setBool(const std::string &name, bool value) const;
	void setInt(const std::string &name, int value) const;
	void setFloat(const std::string &name, float value) const;
	void setVec2(const std::string &name, const glm::vec2 &value) const;
	void setVec2(const std::string &name, float x, float y) const;
	void setVec3(const std::string &name, const glm::vec3 &value) const;
	void setVec3(const std::string &name, float x, float y, float z) const;
	void setVec4(const std::string &name, const glm::vec4 &value) const;
	void setVec4(const std::string &name, float x, float y, float z, float w);
	void setMat2(const std::string &name, const glm::mat2 &mat) const;
	void setMat3(const std::string &name, const glm::mat3 &mat) const;
	void setMat4(const std::string &name, const glm::mat4 &mat) const;

private:
	void checkCompileErrors(GLuint shader, std::string type);
};

//*******************
// CgObject container to be displayed in Viewer
//*******************
class CgObject {
public:
	CgObject();
	~CgObject() {};
	int objectIndex;
	GLuint vao, vbo, ebo;
	GLuint tex1;
	GLfloat* vertexBuffer;
	GLuint* indexBuffer;
	int vertexBufferSize;
	int indexBufferSize;
	int nVertices;
	int nIndices;
	GLsizei nTriangles; //nTriangles = nIndices??

	// Current pose
	float rx, ry, rz;
	float tx, ty, tz;
	// Quaternion rotation
	glm::quat qrot;

	glm::mat4 model;

						//properties
	bool isTextured;

	enum ArrayFormat {
		VERTEX_NORMAL_TEXTURE,
		VERTEX_TEXTURE
	} arrayFormat;

	enum Mode {
		DEFAULT_MODE,
		TRIANGLES,
		POINTS,
		TRIANGLES_POINTS,
		CULLED_POINTS
	} drawMode;

	enum RenderType {
		DEFAULT_RENDER,
		BLACK,
		VERTEX,
		NORMAL,
		EDGE
	} renderType = RenderType::DEFAULT_RENDER;

	void setRenderType(RenderType renderType);

	Shader *shader; //each object has it's own shader

	void loadShader(const char *vertexShader, const char *fragmentShader);
	void bindShader();
	void loadTexture(cv::Mat texture);
	void loadTexture(std::string filename);
	void loadData(std::vector<float> vertices, std::vector<unsigned int> indices);
	void loadData(std::vector<float> vertices, std::vector<unsigned int> indices, ArrayFormat arrayFormat);

	void bindTexture();
	void setMVP(glm::mat4 model, glm::mat4 view, glm::mat4 projection);
	void setNormalMatrix(glm::mat4 model, glm::mat4 view);
	void setLight();
	void setColor(cv::Scalar color, double alpha);
	void setColor(cv::Scalar color);
	void setDrawMode(Mode mode);
	void draw();
	void bindBuffer(); //bind shader and vao
	void release(); //release shader and vao
};

//*******************
// Viewer class
//*******************
class Viewer
{
public:
	Viewer();
	~Viewer();
	void run();
	bool isRunning = false;
	void runOnce();
	void captureDepth(std::string filename);
	void depthAndVertexCapture();
	void depthEdgeAndVertexCapture();
	enum DepthAndVertexCaptureState {
		DEPTH,
		VERTEX,
	} depthAndVertexCaptureState;

	enum ProjectionType {
		DEFAULT_PROJECTION,
		ORTHOGRAPHIC,
		PERSPECTIVE
	} projectionType;

	void setCameraProjectionType(ProjectionType projectionType);
	glm::mat4 projectionMat;

	void close();
	void makeCurrent();
	int createWindow(int scrWidth, int scrHeight, const char *windowName);
	int createWindow(int scrWidth, int scrHeight, const char *windowName, GLFWwindow * existingWindow);
	void processInput(GLFWwindow *window);

	static void framebufferSizeCallback(GLFWwindow* window, int width, int height);
	static void mouseCallback(GLFWwindow* window, double xpos, double ypos);
	static void scrollCallback(GLFWwindow* window, double xoffset, double yoffset);
	static void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods);
	static void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);

	Camera *camera;
	GLFWwindow* window;

	std::vector<CgObject*> *cgObject;

	unsigned int scrWidth, scrHeight;

	//glm::vec3 cameraPos;
	//glm::vec3 cameraFront;
	//glm::vec3 cameraUp;

	float lastX;
	float lastY;
	bool firstMouse;

	// timing
	float deltaTime;	// time between current frame and last frame
	float lastFrame;

	// mouse button states
	bool lButton_down;
	bool rButton_down;
	bool mButton_down;

	//framebuffer
	unsigned int rbo;
	unsigned int depthBuffer; //framebuffer
	unsigned int textureDepthBuffer;
	unsigned int colorBuffer; //framebuffer
	unsigned int textureColorBuffer;
	void setFramebuffer();

	//Capturing depth and vertex images
	void saveFramebuffer(std::string filename);
	void saveDepthAndVertex(std::string depthFileName, std::string vertexFileName,
		std::string previewFileName, std::string normalVectorFilename);
	void saveDepthEdgeAndVertex(std::string depthFileName, std::string vertexFileName, std::string edgeFileName,
		std::string previewFileName, std::string normalVectorFilename);
	std::string depthFileNameHeader;
	std::string previewFileNameHeader;
	std::string vertexFileNameHeader;
	std::string normalFileNameHeader;
	std::string edgeFileNameHeader;
	int depthFileNameCounter;
	int vertexFileNameCounter;
	int edgeFileNameCounter;
	int normalFileNameCounter;
	bool vPressStatus;
	bool captureStatus;
};
