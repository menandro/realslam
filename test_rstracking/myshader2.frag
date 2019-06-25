#version 330 core
out vec4 FragColor;

in vec2 TexCoord;
in vec3 Normal;
in vec3 FragPos;

// texture samplers
uniform sampler2D tex1;
uniform vec3 lightPos;
uniform vec3 lightColor;

void main()
{
	float ambientStrength = 0.1;
    vec3 ambient = ambientStrength * lightColor;

	vec3 norm = normalize(Normal);
	vec3 lightDir = normalize(vec3(1.0f, 1.0f, 0.0f));
	//normalize(lightPos - FragPos);  

	float diff = max(dot(norm, lightDir), 0.0);
	vec3 diffuse = diff * lightColor;;

	vec4 texColor = texture(tex1, TexCoord);
	vec3 objectColor;
	objectColor.x = texColor.x;
	objectColor.y = texColor.y;
	objectColor.z = texColor.z;

	vec3 result = (ambient + diffuse) * objectColor;
	
	FragColor = vec4(result, 1.0);

    //FragColor = texture(tex1, TexCoord);
	
	//FragColor = vec4(Normal, 1.0f);

	//FragColor = vec4(0.0f, 0.0f, 0.0f, 1.0f);
    //FragColor = vec4(ourColor, 1.0);
	//linearly interpolate between both textures (80% container, 20% awesomeface)
	//FragColor = mix(texture(tex1, TexCoord), texture(tex1, TexCoord), 0.2);
}