#version 330 core
out vec4 FragColor;

in vec2 TexCoord;
in vec3 Normal;

// texture samplers
uniform sampler2D tex1;

void main()
{
    FragColor = texture(tex1, TexCoord);
	//FragColor = vec4(Normal, 1.0f);

	//FragColor = vec4(0.0f, 0.0f, 0.0f, 1.0f);
    //FragColor = vec4(ourColor, 1.0);
	//linearly interpolate between both textures (80% container, 20% awesomeface)
	//FragColor = mix(texture(tex1, TexCoord), texture(tex1, TexCoord), 0.2);
}