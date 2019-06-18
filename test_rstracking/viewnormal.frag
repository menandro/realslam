#version 330 core
out vec4 FragColor;

in vec2 TexCoord;
in vec3 Normal;

// texture samplers
uniform sampler2D tex1;

void main()
{
	FragColor = vec4(Normal, 1.0f);
}