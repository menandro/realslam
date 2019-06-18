#version 330 core
layout (location = 0) in vec3 aPos;

out vec4 aColor;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

vec4 packFloatToVec4i(const float value)
{
  const vec4 bitSh = vec4(256.0*256.0*256.0, 256.0*256.0, 256.0, 1.0);
  const vec4 bitMsk = vec4(0.0, 1.0/256.0, 1.0/256.0, 1.0/256.0);
  vec4 res = fract(value * bitSh);
  res -= res.xxyz * bitMsk;
  return res;
}

float unpackFloatFromVec4i(const vec4 value)
{
  const vec4 bitSh = vec4(1.0/(256.0*256.0*256.0), 1.0/(256.0*256.0), 1.0/256.0, 1.0);
  return(dot(value, bitSh));
}

void main()
{
	gl_Position = projection * view * model * vec4(aPos, 1.0f);
	float z = gl_Position.z;
	float r, g, b;
	if (z < 0.0f){
		aColor = vec4(0.0f, 0.0f, 0.0f, 1.0f);
	}
	else {
		if (z > (20.0f/3.0f)){
			r = 1.0f;
			g = 1.0f;
			b = (z/10.0f);
		}
		else if ((z <= (20.0f/3.0f)) && (z > (10.0f/3.0f))){
			r = 1.0f;
			g = (z/10.0f);
			b = 0.0f;
		}
		else if ((z <= (10.0f/3.0f)) && (z > 0.0f)){
			r = (z/10.0f);
			g = 0.0f;
			b = 0.0f;
		}
	}
	
	aColor = vec4(b, g, r, 1.0f);
	//aColor = vec3(r/0.33f, g/0.33f, b/0.33f);
	//aColor = 0.1*vec4(gl_Position.z, gl_Position.z, gl_Position.z, 1.0f);
	//vec4 encode = packFloatToVec4i(0.1*gl_Position.z);
	//float decode = unpackFloatFromVec4i(encode);
	//aColor = vec4(decode, decode, decode, 1.0f);
	
}