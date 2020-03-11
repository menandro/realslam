#pragma once

// Class definition for IMU data
// to be used in all fetching and processing

class Gyro {
public:
	float x; //rate(dot) of rotation in x(rx)
	float y;
	float z;
	double ts; //timestamp
	double lastTs;
	double dt;
};

class Accel {
public:
	float x;
	float y;
	float z;
	double ts; //timestamp
	double lastTs;
	double dt;
};

class Quaternion {
public:
	Quaternion() {
		this->w = 1.0f;
		this->x = 0.0f;
		this->y = 0.0f;
		this->z = 0.0f;
	};
	Quaternion(float x, float y, float z, float w) {
		this->x = x;
		this->y = y;
		this->z = z;
		this->w = w;
	}
	float x;
	float y;
	float z;
	float w;
};

class Vector3 {
public:
	Vector3() {
		this->x = 0.0f;
		this->y = 0.0f;
		this->z = 0.0f;
	};
	Vector3(float x, float y, float z) {
		this->x = x;
		this->y = y;
		this->z = z;
	};
	float x;
	float y;
	float z;
};