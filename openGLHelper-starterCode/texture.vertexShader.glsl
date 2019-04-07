#version 150

// input vertex position and texture coordinates
in vec3 pos;
in vec2 texCoord;

// output texture coordinates
out vec2 tc;

// transformation matrices
uniform mat4 modelViewMatrix;
uniform mat4 projectionMatrix;

void main()
{
  // compute the transformed and projected vertex position (into gl_Position)
  gl_Position = projectionMatrix * modelViewMatrix * vec4(pos, 1.0f);

  // pass-through the texture coordinate
  tc = texCoord;
}

