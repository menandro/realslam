#include "FileReader.h"

FileReader::FileReader() {
	scale = 0.01f;
}

void splitSpaces(std::vector<std::string> &val, std::string value) {
	std::string::size_type start = 0;
	std::string::size_type last = value.find_first_of(" ");
	while (last != std::string::npos) {
		if (last > start) {
			val.push_back(value.substr(start, last - start));
		}
		start = ++last;
		last = value.find_first_of(" ", last);
	}
	val.push_back(value.substr(start));
}

void splitSlash(std::vector<unsigned int> &val, std::string value) {
	std::string::size_type start = 0;
	std::string::size_type last = value.find_first_of("/");
	while (last != std::string::npos) {
		if (last > start) {
			val.push_back(std::stoi(value.substr(start, last - start)));
		}
		start = ++last;
		last = value.find_first_of("/", last);
	}
	val.push_back(std::stoi(value.substr(start)));
}

void splitTwo(float &v1, float &v2, std::string value) {
	std::string val1 = value.substr(0, value.find(' '));
	std::string val2 = value.substr(value.find_first_of(" \t") + 1);
	v1 = std::stof(val1);
	v2 = std::stof(val2);
}

void splitThree(float &v1, float &v2, float &v3, std::string value) {
	std::string val1 = value.substr(0, value.find(' '));
	std::string val23 = value.substr(value.find_first_of(" \t") + 1);
	std::string val2 = val23.substr(0, val23.find(' '));
	std::string val3 = val23.substr(val23.find_first_of(" \t") + 1);
	v1 = std::stof(val1);
	v2 = std::stof(val2);
	v3 = std::stof(val3);
}

bool FileReader::fileExists(const std::string& filename)
{
	struct stat buf;
	if (stat(filename.c_str(), &buf) != -1)
	{
		return true;
	}
	return false;
}

int FileReader::readObj(std::string filename) {
	return readObj(filename, ArrayFormat::VERTEX_NORMAL_TEXTURE, 1.0f);
}

int FileReader::readObj(std::string filename, int arrayFormat) {
	return readObj(filename, arrayFormat, 1.0f);
}

int FileReader::readObj(std::string filename, float scale) {
	return readObj(filename, ArrayFormat::VERTEX_NORMAL_TEXTURE, 1.0f);
}

int FileReader::readObj(std::string filename, int arrayFormat, float scale) {
	this->arrayFormat = (ArrayFormat)arrayFormat;
	this->scale = scale;
	if (!fileExists(filename)) {
		std::cout << "Object File does not exist." << std::endl;
		return -1;
	}
	//read obj file
	std::fstream objectFile(filename, std::ios_base::in);

	//read all lines
	std::string line;
	while (std::getline(objectFile, line)) {
		//std::cout << line << std::endl;
		lines.push_back(line);
	}

	//parse
	int nLines = lines.size();
	for (int nLine = 0; nLine < nLines; nLine++) {
		std::string line = lines[nLine];
		std::vector<std::string> val;
		splitSpaces(val, line);

		if (val[0].compare("o") == 0) {
			this->objectName = val[1];
		}
		else if (val[0].compare("v") == 0) {
			this->obj_vertx.push_back(std::stof(val[1]));
			this->obj_verty.push_back(std::stof(val[2]));
			this->obj_vertz.push_back(std::stof(val[3]));
		}
		else if (val[0].compare("vt") == 0) {
			this->obj_u.push_back(std::stof(val[1]));
			this->obj_v.push_back(std::stof(val[2]));
		}
		else if (val[0].compare("vn") == 0) {
			this->obj_normx.push_back(std::stof(val[1]));
			this->obj_normy.push_back(std::stof(val[2]));
			this->obj_normz.push_back(std::stof(val[3]));
		}
		else if (val[0].compare("f") == 0) {
			//triangles
			if (val.size() == 4) {
				for (int k = 1; k <= 3; k++) {
					std::vector<unsigned int> faceval;
					splitSlash(faceval, val[k]);
					this->obj_tri_vertices.push_back(faceval[0]);
					this->obj_tri_uvs.push_back(faceval[1]);
					this->obj_tri_normals.push_back(faceval[2]);
				}
			}
			//or convert quad to triangles 1234 ->123,134
			else if (val.size() == 5) {
				//first triangle //1-2-3
				int tri1[] = { 1, 2, 3 };
				for (int s = 0; s < 3; s++) {
					std::vector<unsigned int> faceval;
					splitSlash(faceval, val[tri1[s]]);
					this->obj_tri_vertices.push_back(faceval[0]);
					this->obj_tri_uvs.push_back(faceval[1]);
					this->obj_tri_normals.push_back(faceval[2]);
					//std::cout << faceval[0] << " " << faceval[1] << " " << faceval[2] << std::endl;
				}
				//secont triangle //1-3-4
				int tri2[] = { 1, 3, 4 };
				for (int s = 0; s < 3; s++) {
					std::vector<unsigned int> faceval;
					splitSlash(faceval, val[tri2[s]]);
					this->obj_tri_vertices.push_back(faceval[0]);
					this->obj_tri_uvs.push_back(faceval[1]);
					this->obj_tri_normals.push_back(faceval[2]);
					//std::cout << faceval[0] << " " << faceval[1] << " " << faceval[2] << std::endl;
				}
			}
		}
	}
	return this->objToArray();
}

int FileReader::objToArray() {
	if (this->arrayFormat == ArrayFormat::VERTEX_TEXTURE) {
		//convert to 3vert,2tex and triangle format
		int nTris = obj_tri_vertices.size() / 3;
		//std::cout << nTris;
		for (int k = 0; k < nTris; k++) {
			//vertex 1
			unsigned int index1 = obj_tri_vertices[3 * k] - 1;
			vertexArray.push_back(scale*obj_vertx[index1]); //x
			vertexArray.push_back(scale*obj_verty[index1]); //y
			vertexArray.push_back(scale*obj_vertz[index1]); //z
			vertexArray.push_back(obj_u[obj_tri_uvs[3 * k] - 1]);
			vertexArray.push_back(obj_v[obj_tri_uvs[3 * k] - 1]);

			//vertex 2
			unsigned int index2 = obj_tri_vertices[3 * k + 1] - 1;
			vertexArray.push_back(scale*obj_vertx[index2]); //x
			vertexArray.push_back(scale*obj_verty[index2]); //y
			vertexArray.push_back(scale*obj_vertz[index2]); //z
			vertexArray.push_back(obj_u[obj_tri_uvs[3 * k + 1] - 1]);
			vertexArray.push_back(obj_v[obj_tri_uvs[3 * k + 1] - 1]);

			//vertex 3
			unsigned int index3 = obj_tri_vertices[3 * k + 2] - 1;
			vertexArray.push_back(scale*obj_vertx[index3]); //x
			vertexArray.push_back(scale*obj_verty[index3]); //y
			vertexArray.push_back(scale*obj_vertz[index3]); //z
			vertexArray.push_back(obj_u[obj_tri_uvs[3 * k + 2] - 1]);
			vertexArray.push_back(obj_v[obj_tri_uvs[3 * k + 2] - 1]);

			/*indexArray.push_back(obj_tri_vertices[3 * k] - 1);
			indexArray.push_back(obj_tri_vertices[3 * k + 1] - 1);
			indexArray.push_back(obj_tri_vertices[3 * k + 2] - 1);*/

			indexArray.push_back(3 * k);
			indexArray.push_back(3 * k + 1);
			indexArray.push_back(3 * k + 2);
		}
	}

	else if (this->arrayFormat == ArrayFormat::VERTEX_NORMAL_TEXTURE) {
		//convert to 3vert, 3norm, 2tex and triangle format
		int nTris = obj_tri_vertices.size() / 3;
		//std::cout << nTris;
		for (int k = 0; k < nTris; k++) {
			//vertex 1
			unsigned int index1 = obj_tri_vertices[3 * k] - 1;
			vertexArray.push_back(scale*obj_vertx[index1]); //x
			vertexArray.push_back(scale*obj_verty[index1]); //y
			vertexArray.push_back(scale*obj_vertz[index1]); //z

			vertexArray.push_back(obj_normx[obj_tri_normals[3 * k] - 1]); //nx
			vertexArray.push_back(obj_normy[obj_tri_normals[3 * k] - 1]); //ny
			vertexArray.push_back(obj_normz[obj_tri_normals[3 * k] - 1]); //nz

			vertexArray.push_back(obj_u[obj_tri_uvs[3 * k] - 1]);
			vertexArray.push_back(obj_v[obj_tri_uvs[3 * k] - 1]);

			//vertex 2
			unsigned int index2 = obj_tri_vertices[3 * k + 1] - 1;
			vertexArray.push_back(scale*obj_vertx[index2]); //x
			vertexArray.push_back(scale*obj_verty[index2]); //y
			vertexArray.push_back(scale*obj_vertz[index2]); //z

			vertexArray.push_back(obj_normx[obj_tri_normals[3 * k + 1] - 1]); //nx
			vertexArray.push_back(obj_normy[obj_tri_normals[3 * k + 1] - 1]); //ny
			vertexArray.push_back(obj_normz[obj_tri_normals[3 * k + 1] - 1]); //nz

			vertexArray.push_back(obj_u[obj_tri_uvs[3 * k + 1] - 1]);
			vertexArray.push_back(obj_v[obj_tri_uvs[3 * k + 1] - 1]);

			//vertex 3
			unsigned int index3 = obj_tri_vertices[3 * k + 2] - 1;
			vertexArray.push_back(scale*obj_vertx[index3]); //x
			vertexArray.push_back(scale*obj_verty[index3]); //y
			vertexArray.push_back(scale*obj_vertz[index3]); //z

			vertexArray.push_back(obj_normx[obj_tri_normals[3 * k + 2] - 1]); //nx
			vertexArray.push_back(obj_normy[obj_tri_normals[3 * k + 2] - 1]); //ny
			vertexArray.push_back(obj_normz[obj_tri_normals[3 * k + 2] - 1]); //nz

			vertexArray.push_back(obj_u[obj_tri_uvs[3 * k + 2] - 1]);
			vertexArray.push_back(obj_v[obj_tri_uvs[3 * k + 2] - 1]);

			/*indexArray.push_back(obj_tri_vertices[3 * k] - 1);
			indexArray.push_back(obj_tri_vertices[3 * k + 1] - 1);
			indexArray.push_back(obj_tri_vertices[3 * k + 2] - 1);*/

			indexArray.push_back(3 * k);
			indexArray.push_back(3 * k + 1);
			indexArray.push_back(3 * k + 2);
		}
	}
	return 0;
}

int FileReader::splitLines(std::fstream objectFile) {
	return 0;
}

int FileReader::setScale(float scale) {
	this->scale = scale;
	return 0;
}

//std::string prefix = line.substr(0, line.find(' '));
//std::string value = line.substr(line.find_first_of(" \t") + 1);
//float v1, v2, v3, vt1, vt2, vn1, vn2, vn3;
//std::vector<std::string> val;
//
//if (prefix.compare("o") == 0) {
//	this->objectName = value;
//}
//else if (prefix.compare("v") == 0) {
//	//split three values
//	//splitThree(v1, v2, v3, value);
//	splitSpaces(val, value);
//}
//else if (prefix.compare("vt") == 0) {
//	//split three values
//	//splitTwo(v1, v2, value);
//	splitSpaces(val, value);
//}
//else if (prefix.compare("vn") == 0) {
//	//split three values
//	splitThree(v1, v2, v3, value);
//}
//else if (prefix.compare("f") == 0) {
//	//split four values (quad)
//	std::string f1, f2, f3, f4;
//	splitSpaces(val, value);
//	//convert to triangle
//}