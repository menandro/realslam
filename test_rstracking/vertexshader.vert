#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec2 aTexCoord;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main()
{
	//gl_Position = projection * view * model * vec4(aPos, 1.0f);
	vec4 v = view * model * vec4(aPos, 1.0f);
	//v.xyz = v.xyz * 0.999;
	v.z = v.z*0.995;
	gl_Position = projection * v;
	gl_PointSize = 1.0f;
}