#include "Viewer.h"

Viewer::Viewer()
{
	camera = new Camera(glm::vec3(0.0f, 1.0f, 5.0f));
	projectionType = ProjectionType::DEFAULT_PROJECTION;
	camera->orthoZoom = 1.0f;

	//initialize objects to be displayed
	cgObject = new std::vector<CgObject*>();

	//cameraPos = glm::vec3(0.0f, 0.0f, 3.0f);
	//cameraFront = glm::vec3(0.0f, 0.0f, -1.0f);
	//cameraUp = glm::vec3(0.0f, 1.0f, 0.0f);

	deltaTime = 0.0f;	// time between current frame and last frame
	lastFrame = 0.0f;
}

void Viewer::setCameraProjectionType(ProjectionType projectionType) {
	this->projectionType = projectionType;
	if (projectionType == ProjectionType::ORTHOGRAPHIC) {
		float zoom = camera->orthoZoom;
		projectionMat = glm::ortho(-zoom, zoom, -zoom, zoom, 0.001f, 10.0f);
	}
	else if (projectionType == ProjectionType::PERSPECTIVE) {
		projectionMat = glm::perspective(glm::radians(45.0f), (float)this->scrWidth / (float)this->scrHeight, 0.001f, 10.0f);
	}
	else {
		projectionMat = glm::perspective(glm::radians(45.0f), (float)this->scrWidth / (float)this->scrHeight, 0.001f, 10.0f);
	}
}

void Viewer::makeCurrent() {
	glfwMakeContextCurrent(window);
}

int Viewer::createWindow(int scrWidth, int scrHeight, const char *windowName)
{
	glfwInit();
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	// glfw window creation
	// --------------------
	this->scrWidth = scrWidth;
	this->scrHeight = scrHeight;
	this->lastX = scrWidth / 2.0f;
	this->lastY = scrHeight / 2.0f;
	this->firstMouse = true;

	window = glfwCreateWindow(scrWidth, scrHeight, windowName, NULL, NULL);
	if (window == NULL)
	{
		std::cout << "Failed to create GLFW window" << std::endl;
		glfwTerminate();
		return -1;
	}

	glfwMakeContextCurrent(window);
	glfwSetWindowUserPointer(window, this);
	glfwSetFramebufferSizeCallback(window, &Viewer::framebufferSizeCallback);
	glfwSetCursorPosCallback(window, &Viewer::mouseCallback);
	glfwSetMouseButtonCallback(window, &Viewer::mouseButtonCallback);
	glfwSetScrollCallback(window, &Viewer::scrollCallback);
	glfwSetKeyCallback(window, &Viewer::keyCallback);

	// glad: load all OpenGL function pointers
	// ---------------------------------------
	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
	{
		std::cout << "Failed to initialize GLAD" << std::endl;
		return -1;
	}
	glEnable(GL_DEPTH_TEST);
}


Viewer::~Viewer()
{
}

void Viewer::depthAndVertexCapture() {
	makeCurrent();
	this->depthAndVertexCaptureState = this->DEPTH;
	while (!glfwWindowShouldClose(window))
	{
		float currentFrame = glfwGetTime();
		deltaTime = currentFrame - lastFrame;
		lastFrame = currentFrame;

		// input
		// -----
		processInput(window);

		// render
		// ------
		//glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
		//glBindFramebuffer(GL_FRAMEBUFFER, depthBuffer);
		glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		//get camera transformation
		glm::mat4 projection, view, model;
		projection = projectionMat;
		view = camera->GetViewMatrix();
		model = glm::translate(glm::mat4(1.0f), glm::vec3(0.0, 0.0, 0.0));
		// render all cgobjects
		for (std::vector<CgObject*>::iterator it = cgObject->begin(); it != cgObject->end(); ++it) {
			if ((*it)->renderType == CgObject::RenderType::DEFAULT_RENDER) {
				(*it)->bindTexture();
				(*it)->bindShader();
				(*it)->setMVP(model, view, projection);
				(*it)->bindBuffer();
				(*it)->draw();
			}
		}

		// glfw: swap buffers and poll IO events (keys pressed/released, mouse moved etc.)
		// -------------------------------------------------------------------------------
		glfwSwapBuffers(window);
		glfwPollEvents();
	}
}

void Viewer::depthEdgeAndVertexCapture() {
	makeCurrent();
	this->depthAndVertexCaptureState = this->DEPTH;
	while (!glfwWindowShouldClose(window))
	{
		float currentFrame = glfwGetTime();
		deltaTime = currentFrame - lastFrame;
		lastFrame = currentFrame;

		// input
		// -----
		processInput(window);

		// render
		// ------
		//glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
		//glBindFramebuffer(GL_FRAMEBUFFER, depthBuffer);
		glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		//get camera transformation
		glm::mat4 projection, view, model;
		projection = projectionMat;
		view = camera->GetViewMatrix();
		model = glm::translate(glm::mat4(1.0f), glm::vec3(0.0, 0.0, 0.0));
		// render all cgobjects
		for (std::vector<CgObject*>::iterator it = cgObject->begin(); it != cgObject->end(); ++it) {
			if ((*it)->renderType == CgObject::RenderType::DEFAULT_RENDER) {
				(*it)->bindTexture();
				(*it)->bindShader();
				(*it)->setMVP(model, view, projection);
				(*it)->bindBuffer();
				(*it)->draw();
			}
		}

		// glfw: swap buffers and poll IO events (keys pressed/released, mouse moved etc.)
		// -------------------------------------------------------------------------------
		glfwSwapBuffers(window);
		glfwPollEvents();
	}
}

void Viewer::run() {
	makeCurrent();
	while (!glfwWindowShouldClose(window))
	{
		float currentFrame = glfwGetTime();
		deltaTime = currentFrame - lastFrame;
		lastFrame = currentFrame;

		// input
		// -----
		processInput(window);

		// render
		// ------
		//glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
		glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		//get camera transformation
		glm::mat4 projection, view, model;
		projection = projectionMat;
		view = camera->GetViewMatrix();
		model = glm::translate(glm::mat4(1.0f), glm::vec3(0.0, 0.0, 0.0));

		// Call registered Update functions

		// render all cgobjects
		for (std::vector<CgObject*>::iterator it = cgObject->begin(); it != cgObject->end(); ++it) {
			(*it)->bindTexture();
			(*it)->bindShader();
			(*it)->setMVP(model, view, projection);
			(*it)->bindBuffer();
			(*it)->draw();
		}

		// glfw: swap buffers and poll IO events (keys pressed/released, mouse moved etc.)
		// -------------------------------------------------------------------------------
		glfwSwapBuffers(window);
		glfwPollEvents();
	}
}

void Viewer::setFramebuffer() {
	//set depth framebuffer
	glGenFramebuffers(1, &depthBuffer);
	glBindFramebuffer(GL_FRAMEBUFFER, depthBuffer);

	glGenTextures(1, &textureDepthBuffer);
	glBindTexture(GL_TEXTURE_2D, textureDepthBuffer);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH24_STENCIL8, scrWidth, scrHeight, 0, GL_DEPTH_STENCIL, GL_UNSIGNED_INT_24_8, NULL);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, textureDepthBuffer, 0);

	//set color framebuffer
	glGenFramebuffers(1, &colorBuffer);
	glBindFramebuffer(GL_FRAMEBUFFER, colorBuffer);

	glGenTextures(1, &textureColorBuffer);
	glBindTexture(GL_TEXTURE_2D, textureColorBuffer);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, scrWidth, scrHeight, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, textureColorBuffer, 0);

	//glGenRenderbuffers(1, &rbo);
	//glBindRenderbuffer(GL_RENDERBUFFER, rbo);
	//// use a single renderbuffer object for both a depth AND stencil buffer.
	//glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT32F, scrWidth, scrHeight); 
	//glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, rbo); // now actually attach it
	//// now that we actually created the framebuffer and added all attachments we want to check if it is actually complete now
	if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
		std::cout << "ERROR::FRAMEBUFFER:: Framebuffer is not complete!" << std::endl;
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void Viewer::saveFramebuffer(std::string filename) {
	cv::Mat img(scrHeight, scrWidth, CV_8UC4);
	//std::cout << img.elemSize() << std::endl;
	//std::cout << (img.step & 4) ? 1 : 4;
	//glPixelStorei(GL_PACK_ALIGNMENT, (img.step & 3) ? 1 : 4); //TODO::WHAT'S THIS??
	glPixelStorei(GL_PACK_ROW_LENGTH, img.step / img.elemSize());
	//glReadPixels(0, 0, img.cols, img.rows, GL_BGR, GL_UNSIGNED_BYTE, img.data);
	glReadPixels(0, 0, img.cols, img.rows, GL_DEPTH_STENCIL, GL_UNSIGNED_INT_24_8, img.data);
	cv::Mat flipped;
	cv::flip(img, flipped, 0);
	std::vector<cv::Mat> argb; //bgra
	cv::split(flipped, argb);
	cv::Mat alpha = cv::Mat::ones(flipped.size(), CV_8U);
	std::vector<cv::Mat> bgra = { argb[3], argb[2], argb[1], alpha };
	//cv::imshow("red", argb[3]);
	//cv::imshow("green", argb[2]);
	//cv::imshow("blue", argb[1]);
	//cv::imshow("alpha", argb[0]);
	//cv::waitKey();
	cv::Mat bgraout;
	cv::merge(bgra, bgraout);
	cv::imshow("depth", bgraout);
	cv::imwrite(filename, bgraout);
	//glBindTexture(GL_TEXTURE_2D, textureColorbuffer);
	//glReadBuffer(GL_BACK); // Ensure we are reading from the back buffer.
	//glCopyTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, 0, 0, scrWidth, scrHeight, 0);
}

void Viewer::close() {
	glfwTerminate();
}

//callbacks
void Viewer::processInput(GLFWwindow *window)
{
	if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
		camera->ProcessKeyboard(FORWARD, deltaTime);
	if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
		camera->ProcessKeyboard(BACKWARD, deltaTime);
	if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
		camera->ProcessKeyboard(LEFT, deltaTime);
	if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
		camera->ProcessKeyboard(RIGHT, deltaTime);
	//if (glfwGetKey(window, GLFW_KEY_C) == GLFW_PRESS)
	//	saveFramebuffer("screenshot.png");
}

void Viewer::keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
	Viewer* viewer = (Viewer*)glfwGetWindowUserPointer(window);
	if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
		glfwSetWindowShouldClose(window, true);
	if (glfwGetKey(window, GLFW_KEY_PERIOD) == GLFW_PRESS) {
		viewer->camera->resetCameraPosition();
		if (viewer->projectionType == Viewer::ProjectionType::ORTHOGRAPHIC) {
			float zoom = viewer->camera->orthoZoom;
			viewer->projectionMat = glm::ortho(-zoom, zoom, -zoom, zoom, 0.001f, 10.0f);
		}
	}
	if (glfwGetKey(window, GLFW_KEY_V) == GLFW_PRESS) {
		std::string df = viewer->depthFileNameHeader + std::to_string(viewer->depthFileNameCounter) + ".png";
		std::string pf = viewer->previewFileNameHeader + std::to_string(viewer->depthFileNameCounter) + ".png";
		std::string vf = viewer->vertexFileNameHeader + std::to_string(viewer->vertexFileNameCounter) + ".png";
		std::string nf = viewer->normalFileNameHeader + std::to_string(viewer->normalFileNameCounter) + ".png";
		viewer->saveDepthAndVertex(df, vf, pf, nf);
		viewer->depthFileNameCounter++;
		viewer->vertexFileNameCounter++;
		viewer->normalFileNameCounter++;
	}

	if (glfwGetKey(window, GLFW_KEY_G) == GLFW_PRESS) {
		std::string df = viewer->depthFileNameHeader + std::to_string(viewer->depthFileNameCounter) + ".png";
		std::string pf = viewer->previewFileNameHeader + std::to_string(viewer->depthFileNameCounter) + ".png";
		std::string vf = viewer->vertexFileNameHeader + std::to_string(viewer->vertexFileNameCounter) + ".png";
		std::string ef = viewer->edgeFileNameHeader + std::to_string(viewer->depthFileNameCounter) + ".png";
		std::string nf = viewer->normalFileNameHeader + std::to_string(viewer->normalFileNameCounter) + ".png";
		viewer->saveDepthEdgeAndVertex(df, vf, ef, pf, nf);
		viewer->depthFileNameCounter++;
		viewer->vertexFileNameCounter++;
		viewer->edgeFileNameCounter++;
		viewer->normalFileNameCounter++;
	}
}

void Viewer::saveDepthAndVertex(std::string depthFileName, std::string vertexFileName,
	std::string previewFileName, std::string normalVectorFilename) {
	//save depth from previous draw
	cv::Mat depth(scrHeight, scrWidth, CV_8UC4);
	glPixelStorei(GL_PACK_ROW_LENGTH, depth.step / depth.elemSize());
	glReadPixels(0, 0, depth.cols, depth.rows, GL_DEPTH_STENCIL, GL_UNSIGNED_INT_24_8, depth.data);
	cv::Mat depthFlipped;
	cv::flip(depth, depthFlipped, 0);
	std::vector<cv::Mat> depthArgb; //bgra
	cv::split(depthFlipped, depthArgb);
	cv::Mat depthAlpha = cv::Mat::ones(depthFlipped.size(), CV_8U);
	std::vector<cv::Mat> depthBgra = { depthArgb[3], depthArgb[2], depthArgb[1], depthAlpha };
	//std::vector<cv::Mat> depthBgra = { depthArgb[3], depthArgb[2], depthArgb[1], depthArgb[0] }; //stencil pala ung alpha
	cv::Mat depthBgraOut;
	cv::merge(depthBgra, depthBgraOut);
	cv::imshow("depth", depthArgb[3]);
	cv::imwrite(depthFileName, depthBgraOut);
	cv::imwrite(previewFileName, depthArgb[3]);

	//render black and vertex and save
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glm::mat4 projection, view, model;
	projection = projectionMat;
	view = camera->GetViewMatrix();
	model = glm::translate(glm::mat4(1.0f), glm::vec3(0.0, 0.0, 0.0));
	for (std::vector<CgObject*>::iterator it = cgObject->begin(); it != cgObject->end(); ++it) {
		if (((*it)->renderType == CgObject::RenderType::VERTEX) || ((*it)->renderType == CgObject::RenderType::BLACK)) {
			(*it)->bindTexture();
			(*it)->bindShader();
			(*it)->setMVP(model, view, projection);
			(*it)->bindBuffer();
			(*it)->draw();
		}
	}
	cv::Mat vertex(scrHeight, scrWidth, CV_8UC3);
	glPixelStorei(GL_PACK_ROW_LENGTH, vertex.step / vertex.elemSize());
	glReadPixels(0, 0, vertex.cols, vertex.rows, GL_BGR, GL_UNSIGNED_BYTE, vertex.data);
	cv::Mat vertexFlipped;
	cv::flip(vertex, vertexFlipped, 0);
	cv::imshow("vertex", vertexFlipped);
	cv::imwrite(vertexFileName, vertexFlipped);


	//render normal vectors and save
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	for (std::vector<CgObject*>::iterator it = cgObject->begin(); it != cgObject->end(); ++it) {
		if ((*it)->renderType == CgObject::RenderType::NORMAL) {
			(*it)->bindTexture();
			(*it)->bindShader();
			(*it)->setMVP(model, view, projection);
			(*it)->bindBuffer();
			(*it)->draw();
		}
	}
	cv::Mat normal(scrHeight, scrWidth, CV_8UC3);
	glPixelStorei(GL_PACK_ROW_LENGTH, vertex.step / vertex.elemSize());
	glReadPixels(0, 0, normal.cols, normal.rows, GL_BGR, GL_UNSIGNED_BYTE, normal.data);
	cv::Mat normalFlipped;
	cv::flip(normal, normalFlipped, 0);
	cv::imshow("normal", normalFlipped);
	cv::imwrite(normalVectorFilename, normalFlipped);
}

void Viewer::saveDepthEdgeAndVertex(std::string depthFileName, std::string vertexFileName, std::string edgeFileName,
	std::string previewFileName, std::string normalVectorFilename) {
	//save depth from previous draw
	cv::Mat depth(scrHeight, scrWidth, CV_8UC4);
	glPixelStorei(GL_PACK_ROW_LENGTH, depth.step / depth.elemSize());
	glReadPixels(0, 0, depth.cols, depth.rows, GL_DEPTH_STENCIL, GL_UNSIGNED_INT_24_8, depth.data);
	cv::Mat depthFlipped;
	cv::flip(depth, depthFlipped, 0);
	std::vector<cv::Mat> depthArgb; //bgra
	cv::split(depthFlipped, depthArgb);
	cv::Mat depthAlpha = cv::Mat::ones(depthFlipped.size(), CV_8U);
	std::vector<cv::Mat> depthBgra = { depthArgb[3], depthArgb[2], depthArgb[1], depthAlpha };
	//std::vector<cv::Mat> depthBgra = { depthArgb[3], depthArgb[2], depthArgb[1], depthArgb[0] }; //stencil pala ung alpha
	cv::Mat depthBgraOut;
	cv::merge(depthBgra, depthBgraOut);
	cv::imshow("depth", depthArgb[3]);
	cv::imwrite(depthFileName, depthBgraOut);
	cv::imwrite(previewFileName, depthArgb[3]);

	//render black and vertex and save
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glm::mat4 projection, view, model;
	projection = projectionMat;
	view = camera->GetViewMatrix();
	model = glm::translate(glm::mat4(1.0f), glm::vec3(0.0, 0.0, 0.0));
	for (std::vector<CgObject*>::iterator it = cgObject->begin(); it != cgObject->end(); ++it) {
		if (((*it)->renderType == CgObject::RenderType::VERTEX) || ((*it)->renderType == CgObject::RenderType::BLACK)) {
			(*it)->bindTexture();
			(*it)->bindShader();
			(*it)->setMVP(model, view, projection);
			(*it)->bindBuffer();
			(*it)->draw();
		}
	}
	cv::Mat vertex(scrHeight, scrWidth, CV_8UC3);
	glPixelStorei(GL_PACK_ROW_LENGTH, vertex.step / vertex.elemSize());
	glReadPixels(0, 0, vertex.cols, vertex.rows, GL_BGR, GL_UNSIGNED_BYTE, vertex.data);
	cv::Mat vertexFlipped;
	cv::flip(vertex, vertexFlipped, 0);
	cv::imshow("vertex", vertexFlipped);
	cv::imwrite(vertexFileName, vertexFlipped);

	//render black and edges and save
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	for (std::vector<CgObject*>::iterator it = cgObject->begin(); it != cgObject->end(); ++it) {
		if (((*it)->renderType == CgObject::RenderType::EDGE) || ((*it)->renderType == CgObject::RenderType::BLACK)) {
			(*it)->bindTexture();
			(*it)->bindShader();
			(*it)->setMVP(model, view, projection);
			(*it)->bindBuffer();
			(*it)->draw();
		}
	}
	cv::Mat edge(scrHeight, scrWidth, CV_8UC3);
	glPixelStorei(GL_PACK_ROW_LENGTH, edge.step / edge.elemSize());
	glReadPixels(0, 0, edge.cols, edge.rows, GL_BGR, GL_UNSIGNED_BYTE, edge.data);
	cv::Mat edgeFlipped;
	cv::flip(edge, edgeFlipped, 0);
	cv::imshow("edge", edgeFlipped);
	cv::imwrite(edgeFileName, edgeFlipped);

	//render normal vectors and save
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	for (std::vector<CgObject*>::iterator it = cgObject->begin(); it != cgObject->end(); ++it) {
		if ((*it)->renderType == CgObject::RenderType::NORMAL) {
			(*it)->bindTexture();
			(*it)->bindShader();
			(*it)->setMVP(model, view, projection);
			(*it)->bindBuffer();
			(*it)->draw();
		}
	}
	cv::Mat normal(scrHeight, scrWidth, CV_8UC3);
	glPixelStorei(GL_PACK_ROW_LENGTH, vertex.step / vertex.elemSize());
	glReadPixels(0, 0, normal.cols, normal.rows, GL_BGR, GL_UNSIGNED_BYTE, normal.data);
	cv::Mat normalFlipped;
	cv::flip(normal, normalFlipped, 0);
	cv::imshow("normal", normalFlipped);
	cv::imwrite(normalVectorFilename, normalFlipped);
}

void Viewer::framebufferSizeCallback(GLFWwindow* window, int width, int height)
{
	glViewport(0, 0, width, height);
}

void Viewer::mouseCallback(GLFWwindow* window, double xpos, double ypos)
{
	Viewer* viewer = (Viewer*)glfwGetWindowUserPointer(window);
	int rButton_state = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT);
	if (rButton_state == GLFW_PRESS) {
		viewer->rButton_down = true;
	}
	else if (rButton_state == GLFW_RELEASE) {
		viewer->rButton_down = false;
	}

	if (viewer->rButton_down) {
		/*if (viewer->firstMouse)
		{
		viewer->lastX = xpos;
		viewer->lastY = ypos;
		viewer->firstMouse = false;
		}*/

		float xoffset = xpos - viewer->lastX;
		float yoffset = viewer->lastY - ypos; // reversed since y-coordinates go from bottom to top

		viewer->lastX = xpos;
		viewer->lastY = ypos;

		viewer->camera->ProcessMouseMovement(xoffset, yoffset);
	}
}

void Viewer::scrollCallback(GLFWwindow* window, double xoffset, double yoffset)
{
	Viewer* viewer = (Viewer*)glfwGetWindowUserPointer(window);
	if (viewer->projectionType == Viewer::ProjectionType::ORTHOGRAPHIC) {
		viewer->camera->ProcessMouseScrollOrthographic(yoffset);
		float zoom = viewer->camera->orthoZoom;
		viewer->projectionMat = glm::ortho(-zoom, zoom, -zoom, zoom, 0.001f, 10.0f);
	}
	else {
		viewer->camera->ProcessMouseScroll(yoffset);
	}
}

void Viewer::mouseButtonCallback(GLFWwindow* window, int button, int action, int mods)
{
	Viewer* viewer = (Viewer*)glfwGetWindowUserPointer(window);

	if ((button == GLFW_MOUSE_BUTTON_RIGHT) && (action == GLFW_PRESS)) {
		double xpos, ypos;
		glfwGetCursorPos(window, &xpos, &ypos);
		viewer->lastX = xpos;
		viewer->lastY = ypos;
	}

	//if (button == GLFW_MOUSE_BUTTON_LEFT) {
	//	if (GLFW_PRESS == action)
	//		viewer->lButton_down = true;
	//	else if (GLFW_RELEASE == action)
	//		viewer->lButton_down = false;
	//}

	//if (viewer->lButton_down){
	//	double xpos, ypos;
	//	glfwGetCursorPos(window, &xpos, &ypos);

	//	if (viewer->firstMouse)
	//	{
	//		viewer->lastX = xpos;
	//		viewer->lastY = ypos;
	//		viewer->firstMouse = false;
	//	}

	//	float xoffset = xpos - viewer->lastX;
	//	float yoffset = viewer->lastY - ypos; // reversed since y-coordinates go from bottom to top

	//	viewer->lastX = xpos;
	//	viewer->lastY = ypos;

	//	viewer->camera->ProcessMouseMovement(xoffset, yoffset);
	//}		
}


//***************************************
// USELESS FUNCTIONS
void Viewer::runOnce() {
	makeCurrent();
	//glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
	if (!glfwWindowShouldClose(window))
	{
		// render
		// ------
		glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		//get camera transformation
		glm::mat4 projection, view, model;
		projection = glm::perspective(glm::radians(camera->Zoom), (float)this->scrWidth / (float)this->scrHeight, 0.1f, 100.0f);
		view = camera->GetViewMatrix();
		model = glm::translate(glm::mat4(1.0f), glm::vec3(0.0, 0.0, 0.0));
		// render all cgobjects
		for (std::vector<CgObject*>::iterator it = cgObject->begin(); it != cgObject->end(); ++it) {
			(*it)->bindTexture();
			(*it)->bindShader();
			(*it)->setMVP(model, view, projection);
			(*it)->bindBuffer();
			(*it)->draw();
		}

		//READ FRAME BUFFER
		saveFramebuffer("test.png");

		glfwSwapBuffers(window);
	}

	while (!glfwWindowShouldClose(window)) {
		glfwPollEvents();
	}
}

void Viewer::captureDepth(std::string filename) {
	makeCurrent();
	//glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
	if (!glfwWindowShouldClose(window))
	{
		// render
		// ------
		glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		//get camera transformation
		glm::mat4 projection, view, model;
		projection = glm::perspective(glm::radians(camera->Zoom), (float)this->scrWidth / (float)this->scrHeight, 0.1f, 100.0f);
		view = camera->GetViewMatrix();
		model = glm::translate(glm::mat4(1.0f), glm::vec3(0.0, 0.0, 0.0));
		// render all cgobjects
		for (std::vector<CgObject*>::iterator it = cgObject->begin(); it != cgObject->end(); ++it) {
			(*it)->bindTexture();
			(*it)->bindShader();
			(*it)->setMVP(model, view, projection);
			(*it)->bindBuffer();
			(*it)->draw();
		}

		//READ FRAME BUFFER
		saveFramebuffer(filename);

		glfwSwapBuffers(window);
	}

	while (!glfwWindowShouldClose(window)) {
		glfwPollEvents();
	}
}

