//=============================================================================================
// Mintaprogram: Zöld háromszög. Ervenyes 2019. osztol.
//
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat, BOM kihuzando.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kiveve
// - Mashonnan atvett programresszleteket forrasmegjeloles nelkul felhasznalni es
// - felesleges programsorokat a beadott programban hagyni!!!!!!! 
// - felesleges kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak 
// A keretben nem szereplo GLUT fuggvenyek tiltottak.
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : ECSGGY
// Neptun : SAGI BENEDEK
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================
#include "framework.h"

// vertex shader in GLSL: It is a Raw string (C++11) since it contains new line characters
const char* const vertexSource = R"(
	#version 330
    precision highp float;

	layout(location = 0) in vec2 cVertexPosition;	// Attrib Array 0
	out vec2 texcoord;

	void main() {
		texcoord = (cVertexPosition + vec2(1, 1))/2;							// -1,1 to 0,1
		gl_Position = vec4(cVertexPosition.x, cVertexPosition.y, 0, 1); 		// transform to clipping space
	}
)";

// fragment shader in GLSL
const char* const fragmentSource = R"(
	#version 330
    precision highp float;

	uniform sampler2D textureUnit;
	in  vec2 texcoord;			// interpolated texture coordinates
	out vec4 fragmentColor;		// output that goes to the raster memory as told by glBindFragDataLocation

	void main() {
		fragmentColor = texture(textureUnit, texcoord); 
	}
)";

struct Material {
	vec3 ka, kd, ks;
	float  shininess;
	Material(vec3 _kd, vec3 _ks, float _shininess) : ka(_kd* M_PI), kd(_kd), ks(_ks) { shininess = _shininess; }
};

struct Hit {
	float t;
	vec3 position, normal;
	Material* material;
	Hit() { t = -1; }
};

struct Ray {
	vec3 start, dir;
	Ray(vec3 _start, vec3 _dir) { start = _start; dir = normalize(_dir); }

};

void rayRriteToConsole(const Ray& ray)
{
	printf("start -> x: %f, y: %f, z: %f\n", ray.start.x, ray.start.y, ray.start.z);
	printf("dir -> x: %f, y: %f, z: %f\n", ray.dir.x, ray.dir.y, ray.dir.z);
}

class Intersectable {
protected:
	Material* material;
public:
	virtual Hit intersect(const Ray& ray) = 0;
};

struct plane {
	std::vector<vec3> points;
	vec3 normalVector;
	/*
	vec3 min;
	vec3 max;
	*/


	/*
	plane(vec3 normalVector0, vec3 point0, vec3 min0, vec3 max0) 
	{	normalVector = normalVector0; 
		point = point0;
		min = min0;
		min = max0;
	}
	*/
	plane( std::vector<vec3> points0)
	{
		points = points0;
		normalVector = cross(points[1] - points[0], points[2] - points[0]);
	}

	bool isInsideArea(vec3 foundPoint)
	{
		vec3 crossOne = cross((points[1] - points[0]), (foundPoint - points[0]));
		vec3 crossTwo = cross((points[2] - points[1]), (foundPoint - points[1]));
		vec3 crossThree = cross((points[3] - points[2]), (foundPoint - points[2]));
		vec3 crossFour = cross((points[0] - points[3]), (foundPoint - points[3]));

		float dotOne = dot(crossOne, normalVector);
		float dotTwo = dot(crossTwo, normalVector);
		float dotThree = dot(crossThree, normalVector);
		float dotFour = dot(crossFour, normalVector);
		if (dotOne > 0 && dotTwo > 0 && dotThree > 0 && dotFour > 0)
			return true;
		return false;
	}
};

struct RectangleOwn : public Intersectable {
	std::vector<plane> planes;

	RectangleOwn(Material* material0)
	{	
		planes = std::vector<plane>
		{
			/*
			plane(vec3(1,0,0),vec3(1,0,0),vec3(1,0,0),vec3(1,1,1)),
			plane(vec3(-1,0,0),vec3(0,0,0),vec3(0,0,0), vec3(0,1,1)),
			plane(vec3(0,1,0),vec3(0,1,0),vec3(0,1,0),vec3(1,1,1)),
			plane(vec3(0,-1,0),vec3(0,0,0),vec3(0,0,0), vec3(1,0,1)),
			plane(vec3(0,0,1),vec3(0,0,1),vec3(0,0,1), vec3(1,1,1)),
			plane(vec3(0,0,-1),vec3(0,0,0),vec3(0,0,0),vec3(1,1,0))
			*/
			plane(std::vector<vec3>{vec3(1,0,0),vec3(1,1,0),vec3(1,1,1),vec3(1,0,1)}),
			plane(std::vector<vec3>{vec3(0,0,0),vec3(0,1,0),vec3(0,1,1),vec3(0,0,1)}),
			plane(std::vector<vec3>{vec3(0,1,0),vec3(0,1,1),vec3(1,1,1),vec3(1,1,0)}),
			plane(std::vector<vec3>{vec3(0,0,0),vec3(0,0,1),vec3(1,0,1),vec3(1,0,0)}),
			plane(std::vector<vec3>{vec3(0,0,1),vec3(0,1,1),vec3(1,1,1),vec3(0,1,1)}),
			plane(std::vector<vec3>{vec3(0,0,0),vec3(0,1,0),vec3(1,1,0),vec3(0,1,0)})
		};
		
		material = material0;
		rectangleCrease(1, 0);

	}
	void rectangleCrease(float s, float d)
	{
		
	}


	Hit intersect(const Ray& ray)
	{
		Hit hit;
		vec3 normal(0, 0, 0);
		float temptT = -INFINITY;
		for (int i = 0; i < planes.size(); i++)
		{
			float t = dot(planes[i].points[0] - ray.start, planes[i].normalVector) / dot(ray.dir, planes[i].normalVector);
			if (t > 0) {
				vec3 rayt = ray.start + ray.dir * t;
				vec3 point = rayt - planes[i].points[0];
				if (dot(point, planes[i].normalVector) == 0)
				{
					if (planes[i].isInsideArea(point))
					{
						if (temptT == -INFINITY)
						{ 
							if (temptT < t)
							{
							temptT = t;
							normal = planes[i].normalVector;
							}
						}
						else
						{
							if (temptT > t)
							{
								temptT = t;
								normal = planes[i].normalVector;
							}
						}
					}
				}
			}
		}
		if (temptT < 0) return hit;
		hit.t = temptT;
		hit.material = material;
		hit.normal = normal;
		hit.position = ray.start + ray.dir * hit.t;
		return hit;
	}
};

class Camera {
	vec3 eye, lookat, right, up;
	float fova;
public:
	void set(vec3 _eye, vec3 _lookat, vec3 vup, float fov) {
		eye = _eye;
		lookat = _lookat;
		vec3 w = eye - lookat;
		float focus = length(w);
		right = normalize(cross(vup, w)) * focus * tanf(fov / 2);
		up = normalize(cross(w, right)) * focus * tanf(fov / 2);
		fova = fov;
	}
	Ray getRay(int X, int Y) {
		vec3 dir = lookat + right * (2.0f * (X + 0.5f) / windowWidth - 1) + up * (2.0f * (Y + 0.5f) / windowHeight - 1) - eye;
		return Ray(eye, dir);
	}
};

struct Light {
	vec3 direction;
	vec3 Le;
	Light(vec3 _direction, vec3 _Le) {
		direction = normalize(_direction);
		Le = _Le;
	}
};

float rnd() { return (float)rand() / RAND_MAX; }

const float epsilon = 0.0001f;


class Scene {
	std::vector<Intersectable*> objects;
	std::vector<Light*> lights;
	Camera camera;
	vec3 La;
public:
	void build() {
		vec3 eye = vec3(4,1,4), vup = vec3(3,3,3), lookat = vec3(3,1,3);
		float fov = 45 * M_PI / 180;
		camera.set(eye, lookat, vup, fov);

		La = vec3(0.1f, 0.1f, 0.1f);
		vec3 lightDirection(1, 1, 1), Le(1, 1, 1);
		lights.push_back(new Light(lightDirection, Le));

		vec3 kd(0.0f, 0.0f, 0.0f), ks(1, 1, 1);
		Material* material = new Material(kd, ks, 0);
		objects.push_back(new RectangleOwn(material));
	}


	void render(std::vector<vec4>& image) {
		for (int Y = 0; Y < windowHeight; Y++) {
#pragma omp parallel for
			for (int X = 0; X < windowWidth; X++) {
				vec3 color = trace(camera.getRay(X, Y));
				image[Y * windowWidth + X] = vec4(color.x, color.y, color.z, 1);
			}
		}
	}

	Hit firstIntersect(Ray ray) {
		Hit bestHit;
		for (Intersectable* object : objects) {
			Hit hit = object->intersect(ray); //  hit.t < 0 if no intersection
			if (hit.t > 0 && (bestHit.t < 0 || hit.t < bestHit.t))  bestHit = hit;
		}
		if (dot(ray.dir, bestHit.normal) > 0) bestHit.normal = bestHit.normal * (-1);
		return bestHit;
	}
	
	bool shadowIntersect(Ray ray) {	// for directional lights
		for (Intersectable* object : objects) if (object->intersect(ray).t > 0) return true;
		return false;
	}

	vec3 trace(Ray ray, int depth = 0) {
		Hit hit = firstIntersect(ray);
		if (hit.t < 0) return La;
		vec3 outRadiance = hit.material->ka * La;
		for (Light* light : lights) {
			Ray shadowRay(hit.position + hit.normal * epsilon, light->direction);
			float cosTheta = dot(hit.normal, light->direction);
			if (cosTheta > 0 && !shadowIntersect(shadowRay)) {	// shadow computation
				outRadiance = outRadiance + light->Le * hit.material->kd * cosTheta;
				vec3 halfway = normalize(-ray.dir + light->direction);
				float cosDelta = dot(hit.normal, halfway);
				if (cosDelta > 0) outRadiance = outRadiance + light->Le * hit.material->ks * powf(cosDelta, hit.material->shininess);
			}
		}
		return outRadiance;
	}
};


GPUProgram gpuProgram; // vertex and fragment shaders
Scene scene;
unsigned int vao;	   // virtual world on the GPU

class FullScreenTexturedQuad {
	unsigned int vao;	// vertex array object id and texture id
	Texture texture;
public:
	FullScreenTexturedQuad(int windowWidth, int windowHeight, std::vector<vec4>& image)
		: texture(windowWidth, windowHeight, image)
	{
		glGenVertexArrays(1, &vao);	// create 1 vertex array object
		glBindVertexArray(vao);		// make it active

		unsigned int vbo;		// vertex buffer objects
		glGenBuffers(1, &vbo);	// Generate 1 vertex buffer objects

		// vertex coordinates: vbo0 -> Attrib Array 0 -> vertexPosition of the vertex shader
		glBindBuffer(GL_ARRAY_BUFFER, vbo); // make it active, it is an array
		float vertexCoords[] = { -1, -1,  1, -1,  1, 1,  -1, 1 };	// two triangles forming a quad
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertexCoords), vertexCoords, GL_STATIC_DRAW);	   // copy to that part of the memory which is not modified 
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);     // stride and offset: it is tightly packed
	}

	void Draw() {
		glBindVertexArray(vao);	// make the vao and its vbos active playing the role of the data source
		gpuProgram.setUniform(texture, "textureUnit");
		glDrawArrays(GL_TRIANGLE_FAN, 0, 4);	// draw two triangles forming a quad
	}
};
FullScreenTexturedQuad* fullScreenTexturedQuad;



// Initialization, create an OpenGL context
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
	scene.build();

	std::vector<vec4> image(windowWidth * windowHeight);
	long timeStart = glutGet(GLUT_ELAPSED_TIME);
	scene.render(image);
	long timeEnd = glutGet(GLUT_ELAPSED_TIME);
	printf("Rendering time: %d milliseconds\n", (timeEnd - timeStart));

	// copy image to GPU as a texture
	fullScreenTexturedQuad = new FullScreenTexturedQuad(windowWidth, windowHeight, image);

	// create program for the GPU
	gpuProgram.create(vertexSource, fragmentSource, "fragmentColor");
}

// Window has become invalid: Redraw
void onDisplay() {
	fullScreenTexturedQuad->Draw();
	glutSwapBuffers();
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
	if (key == 'd') glutPostRedisplay();         // if d, invalidate display, i.e. redraw
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {
}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {	// pX, pY are the pixel coordinates of the cursor in the coordinate system of the operation system
	// Convert to normalized device space
	float cX = 2.0f * pX / windowWidth - 1;	// flip y axis
	float cY = 1.0f - 2.0f * pY / windowHeight;
	printf("Mouse moved to (%3.2f, %3.2f)\n", cX, cY);
}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) { // pX, pY are the pixel coordinates of the cursor in the coordinate system of the operation system
	// Convert to normalized device space
	float cX = 2.0f * pX / windowWidth - 1;	// flip y axis
	float cY = 1.0f - 2.0f * pY / windowHeight;

	const char* buttonStat = "";
	switch (state) {
	case GLUT_DOWN: buttonStat = "pressed"; break;
	case GLUT_UP:   buttonStat = "released"; break;
	}

	switch (button) {
	case GLUT_LEFT_BUTTON:   printf("Left button %s at (%3.2f, %3.2f)\n", buttonStat, cX, cY);   break;
	case GLUT_MIDDLE_BUTTON: printf("Middle button %s at (%3.2f, %3.2f)\n", buttonStat, cX, cY); break;
	case GLUT_RIGHT_BUTTON:  printf("Right button %s at (%3.2f, %3.2f)\n", buttonStat, cX, cY);  break;
	}
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {

	long time = glutGet(GLUT_ELAPSED_TIME); // elapsed time since the start of the program
}
