#version 150

// interpolated from vertex program outputs
in vec3 viewPosition;
in vec3 viewNormal;

out vec4 c; // output color

uniform vec4 La; // light ambient
uniform vec4 Ld; // light diffuse
uniform vec4 Ls; // light specular
uniform vec3 viewLightDirection;

uniform vec4 ka; // mesh ambient
uniform vec4 kd; // mesh diffuse
uniform vec4 ks; // mesh specular
uniform float alpha; // shininess

void main()
{
  // camera is at (0,0,0) after the modelview transformation
  vec3 eyedir = normalize(vec3(0, 0, 0) - viewPosition);

  // reflected light direction
  vec3 reflectDir = -reflect(viewLightDirection, viewNormal);

  // Phong lighting
  //float d = max(dot(viewLightDirection, viewNormal), 0.0f);
  //float s = max(dot(reflectDir, eyedir), 0.0f);

  // Cartoon Shading
  // -> use 1D map for L.N and V.R
  //    so that the color of the spline has specific range of variation
  float u = dot(viewLightDirection, viewNormal);
  float d_new, s_new;

  if (u < 0.25) {
	d_new = 0.25; //vec4(0.25, 0.25, 0.25, 1.0);
	s_new = 0.25; //vec4(0.25, 0.25, 0.25, 1.0);
  }
  else if (u < 0.5) {
  	d_new = 0.5; //vec4(0.5, 0.5, 0.5, 1.0);
  	s_new = 0.5; //vec4(0.5, 0.5, 0.5, 1.0);
  }
  else if (u < 0.75) {
  	d_new = 0.75; //vec4(0.75, 0.75, 0.75, 1.0);
  	s_new = 0.75; //vec4(0.75, 0.75, 0.75, 1.0);
  }
  else {
    d_new = 1.0; //vec4(1.0, 1.0, 1.0, 1.0);
    s_new = 1.0; //vec4(1.0, 1.0, 1.0, 1.0);
  }

  // compute the final color
  //c = ka * La + d * kd * Ld + pow(s, alpha) * ks * Ls;
  c = c = ka * La + d_new * kd * Ld + pow(s_new, alpha) * ks * Ls;

  // Check if normal changes as camera moves
  //c.xyz = viewNormal;
}

