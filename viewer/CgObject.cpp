#include "Viewer.h"

CgObject::CgObject() {
	glGenVertexArrays(1, &vao);
	glGenBuffers(1, &vbo);
	glGenBuffers(1, &ebo);
	drawMode = Mode::DEFAULT_MODE;
	isTextured = false;
	renderType = RenderType::DEFAULT_RENDER;
	rx = 0.0f;
	ry = 0.0f;
	rz = 0.0f;
	tx = 0.0f;
	ty = 0.0f;
	tz = 0.0f;
}

void CgObject::bindBuffer() {
	glBindVertexArray(this->vao);
}

void CgObject::loadShader(const char *vertexShader, const char *fragmentShader) {
	shader = new Shader(vertexShader, fragmentShader);
}

void CgObject::loadData(std::vector<float> vertices, std::vector<unsigned int> indices) {
	arrayFormat = ArrayFormat::VERTEX_NORMAL_TEXTURE;
	loadData(vertices, indices, arrayFormat);
}

void CgObject::loadData(std::vector<float> vertices, std::vector<unsigned int> indices, ArrayFormat arrayFormat) {
	nTriangles = indices.size();
	this->bindBuffer();

	if (arrayFormat == ArrayFormat::VERTEX_NORMAL_TEXTURE) {
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float), vertices.data(), GL_STATIC_DRAW);

		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), indices.data(), GL_STATIC_DRAW);

		// position attribute
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)0);
		glEnableVertexAttribArray(0);

		// normal attribute
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(3 * sizeof(float)));
		glEnableVertexAttribArray(1);

		// texture coord attribute
		glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(6 * sizeof(float)));
		glEnableVertexAttribArray(2);
	}

	else if (arrayFormat == ArrayFormat::VERTEX_TEXTURE) {
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float), vertices.data(), GL_STATIC_DRAW);

		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), indices.data(), GL_STATIC_DRAW);

		// position attribute
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
		glEnableVertexAttribArray(0);
		// color attribute
		//glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
		//glEnableVertexAttribArray(1);

		// texture coord attribute
		glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));
		glEnableVertexAttribArray(1);
	}
}

void CgObject::loadTexture(std::string filename) {
	cv::Mat texture = cv::imread(filename);
	this->loadTexture(texture);
}

void CgObject::loadTexture(cv::Mat texture) {
	if (texture.empty()) {
		texture = cv::imread("default_texture.jpg");
	}
	cv::flip(texture, texture, 0);
	isTextured = true;

	glGenTextures(1, &tex1);
	glBindTexture(GL_TEXTURE_2D, tex1);
	// set the texture wrapping parameters
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);	// set texture wrapping to GL_REPEAT (default wrapping method)
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	// set texture filtering parameters
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

	glTexImage2D(GL_TEXTURE_2D,     // Type of texture
		0,                 // Pyramid level (for mip-mapping) - 0 is the top level
		GL_RGB,            // Internal colour format to convert to
		texture.cols,          // Image width  i.e. 640 for Kinect in standard mode
		texture.rows,          // Image height i.e. 480 for Kinect in standard mode
		0,                 // Border width in pixels (can either be 1 or 0)
		GL_BGR, // Input image format (i.e. GL_RGB, GL_RGBA, GL_BGR etc.)
		GL_UNSIGNED_BYTE,  // Image data type
		texture.ptr());        // The actual image data itself

	glGenerateMipmap(GL_TEXTURE_2D);
	shader->use();
	shader->setInt("tex1", 0);
}

void CgObject::bindTexture() {
	if (isTextured) {
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, tex1);
	}
}

void CgObject::bindShader() {
	this->shader->use();
}

void CgObject::setNormalMatrix(glm::mat4 model, glm::mat4 view) {
	glm::mat3x3 normalMatrix = glm::transpose(glm::inverse(glm::mat3(view*model)));
	shader->setMat3("normalMatrix", normalMatrix);
}

void CgObject::setLight() {
	shader->setVec3("lightColor", 1.0f, 1.0f, 1.0f);
	shader->setVec3("lightPos", 1.2f, 1.0f, 2.0f);
}

void CgObject::setColor(cv::Scalar color, double alpha) {
	shader->setVec4("color", (float)color(0), (float)color(1), (float)color(2), (float)alpha);
}

void CgObject::setColor(cv::Scalar color) {
	setColor(color, 1.0);
}

void CgObject::setMVP(glm::mat4 model, glm::mat4 view, glm::mat4 projection) {
	shader->setMat4("mvpMatrix", projection*view*model);
	glm::mat3x3 normalMatrix = glm::transpose(glm::inverse(glm::mat3(view*model)));
	shader->setMat3("normalMatrix", normalMatrix);
	shader->setMat4("projection", projection);
	shader->setMat4("view", view);
	shader->setMat4("model", model);
	//std::cout << glm::to_string(model) << std::endl;
}

void CgObject::setDrawMode(Mode mode) {
	drawMode = mode;
	if ((mode == Mode::POINTS) || (mode == Mode::TRIANGLES_POINTS)) {
		glEnable(GL_PROGRAM_POINT_SIZE);
	}
}

void CgObject::setRenderType(RenderType renderType) {
	this->renderType = renderType;
}

void CgObject::draw() {
	switch (this->drawMode) {
	case Mode::DEFAULT_MODE:
		glDrawElements(GL_TRIANGLES, this->nTriangles, GL_UNSIGNED_INT, 0);
		break;
	case Mode::TRIANGLES:
		glDepthFunc(GL_LESS);
		glDrawElements(GL_TRIANGLES, this->nTriangles, GL_UNSIGNED_INT, 0);
		break;
	case Mode::POINTS:
		//glDepthFunc(GL_LEQUAL);
		glPolygonMode(GL_FRONT_AND_BACK, GL_POINT);
		glDrawElements(GL_TRIANGLES, this->nTriangles, GL_UNSIGNED_INT, 0);
		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
		//glDepthFunc(GL_LESS);
		break;
	case Mode::TRIANGLES_POINTS:
		glDrawElements(GL_TRIANGLES, this->nTriangles, GL_UNSIGNED_INT, 0);
		glDrawElements(GL_POINTS, this->nTriangles, GL_UNSIGNED_INT, 0);
		break;
	case Mode::CULLED_POINTS:
		glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
		glDrawElements(GL_TRIANGLES, this->nTriangles, GL_UNSIGNED_INT, 0);
		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
		break;
	}

	//glDrawArrays(GL_TRIANGLES, 0, 8);
}

void CgObject::release() {

}