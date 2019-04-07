#version 150

// input vertex position and normal in world-space
in vec3 position;
in vec3 normal;

// vertex position and normal, in view-space
out vec3 viewPosition;
out vec3 viewNormal;

// transformation matrices
uniform mat4 modelViewMatrix;
uniform mat4 normalMatrix;
uniform mat4 projectionMatrix;

void main()
{
  // view-space position of the vertex
  vec4 viewPosition4 = modelViewMatrix * vec4(position, 1.0f);
  viewPosition = viewPosition4.xyz;

  // compute the transformed and projected vertex position (into gl_Position)
  // final position in the normalized device coordinates space
  gl_Position = projectionMatrix * viewPosition4;

  // view-space normal
  viewNormal = normalize((normalMatrix*vec4(normal, 0.0f)).xyz);
}

