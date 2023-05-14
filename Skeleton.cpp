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

class Intersectable {
protected:
	Material* material;
public:
	virtual Hit intersect(const Ray& ray) = 0;
};

struct Triangle : public Intersectable
{
public:
	vec3 r1;
	vec3 r2;
	vec3 r3;
	vec3 normalVector;

	Triangle(vec3 _r1, vec3 _r2, vec3 _r3, Material* _material)
	{

		r1 = _r1;
		r2 = _r2;
		r3 = _r3;
		normalVector = cross(r2 - r1, r3 - r1);
		material = _material;
	}

	bool isInsideArea(vec3 p)
	{
		vec3 cross1 = cross(r2 - r1, p - r1);
		vec3 cross2 = cross(r3 - r2, p - r3);
		vec3 cross3 = cross(r1 - r3, p - r3);
		float dot1 = dot(cross1, normalVector);
		float dot2 = dot(cross2, normalVector);
		float dot3 = dot(cross3, normalVector);

		if (dot1 > 0 && dot2 > 0 && dot3 > 0) return true;
		return false;
	}

	Hit intersect(const Ray& ray)
	{
		Hit hit;
		float t = dot(r1 - ray.start, normalVector) / dot(ray.dir, normalVector);
		if (t > 0) 
		{
			vec3 rayt = ray.start + ray.dir * t;
			vec3 point = rayt - r1;
			if (dot(point, normalVector) < 0.1 && dot(point,normalVector) > -0.1)
			{
				if (isInsideArea(point))
				{
					hit.t = t;
					hit.normal = normalVector;
					hit.position = point;
					hit.material = material;
				}
			}
		}
		return hit;	

	}
};
struct RectangleOwn :  Triangle
{
	vec3 r4;
	RectangleOwn(vec3 _r1, vec3 _r2, vec3 _r3, vec3 _r4, Material* _material) : Triangle(_r1,_r2,_r3, _material)
	{
		r4 = _r4;
	}
	bool isInsideArea(vec3 p)
	{
		vec3 cross1 = cross(r2 - r1, p - r1);
		vec3 cross2 = cross(r3 - r2, p - r2);
		vec3 cross3 = cross(r4 - r3, p - r3);
		vec3 cross4 = cross(r1 - r4, p - r4);
		float dot1 = dot(cross1, normalVector);
		float dot2 = dot(cross2, normalVector);
		float dot3 = dot(cross3, normalVector);
		float dot4 = dot(cross4, normalVector);
		if (dot1 > 0 && dot2 > 0 && dot3 > 0 && dot4 > 0) return true;
		return false;
	}
	Hit intersect(const Ray& ray)
	{
		Hit hit;
		float t = dot(r1 - ray.start, normalVector) / dot(ray.dir, normalVector);
		if (t > 0)
		{
			vec3 rayt = ray.start + ray.dir * t;
			vec3 point = rayt - r1;
			if (dot(point, normalVector) < 0.1 && dot(point, normalVector) > -0.1)
			{
				if (isInsideArea(point))
				{
					hit.t = t;
					hit.normal = normalVector;
					hit.position = point;
					hit.material = material;
					return hit;
				}
			}
		}
		return hit;
	}

};
struct Room : Intersectable
{
	std::vector<RectangleOwn> rectangles;
	Room(std::vector<RectangleOwn> rectangles0) { rectangles = rectangles0;}
	Hit intersect(const Ray& ray)
	{
		Hit lowestHit;
		for (int i = 0; i < rectangles.size(); i++)
		{
			Hit hit = rectangles[i].intersect(ray);
			if (hit.t > 0)
				if (lowestHit.t == -1)
					lowestHit = hit;
				else
					if (hit.t > lowestHit.t)
						lowestHit = hit;
		}
		return lowestHit;
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
		float dt = 0;
		vec3 eye = vec3(4, 0.5, 4);
		vec3 vup = vec3(0, 1, 0);
		vec3 lookat = vec3(0, 0, 0);
		eye = vec3((eye.x - lookat.x) * cos(dt) + (eye.z - lookat.z) * sin(dt) + lookat.x,
			eye.y,
			-(eye.x - lookat.x) * sin(dt) + (eye.z - lookat.z) * cos(dt) + lookat.z);
		float fov = 45 * M_PI / 180;
		camera.set(eye, lookat, vup, fov);

		La = vec3(1, 1, 1);
		vec3 lightDirection(1, 1, 1), Le(1, 1, 1);
		lights.push_back(new Light(lightDirection, Le));

		vec3 kd1(0.2, 0.2, 0.2), ks1(1, 1, 1);
		vec3 kd2(0, 0, 0), ks2(0, 0, 0);
		Material* material = new Material(kd1, ks1,1);

		std::vector<vec3> points{
		vec3(0,0,0),
		vec3(1,0,0),
		vec3(0,1,0),
		vec3(0,0,1)
		};
		std::vector<Triangle*> triangles = std::vector<Triangle*>{
		new Triangle(points[0],points[1],points[2], material),
		new Triangle(points[0],points[3],points[2], material),
		new Triangle(points[0],points[1],points[3], material),
		new Triangle(points[1],points[2],points[3], material)
			};
		const vec3 corrig1 = vec3(2.9f, 1.7, 1);
		const float ratio = 4;
		/*
		vec3 dvert1 = (vec3(1, 0, 0) + corrig1) / ratio;
		vec3 dvert2 = (vec3(0, -1, 0) + corrig1) / ratio;
		vec3 dvert3 = (vec3(-1, 0, 0) + corrig1) / ratio;
		vec3 dvert4 = (vec3(0, 1, 0) + corrig1) / ratio;
		vec3 dvert5 = (vec3(0, 0, 1) + corrig1) / ratio;
		vec3 dvert6 = (vec3(0, 0, -1) + corrig1) / ratio;

		Triangle* dface1 = new Triangle(dvert2, dvert1, dvert5, material);
		Triangle* dface2 = new Triangle(dvert3, dvert2, dvert5, material);
		Triangle* dface3 = new Triangle(dvert4, dvert3, dvert5, material);
		Triangle* dface4 = new Triangle(dvert1, dvert4, dvert5, material);
		Triangle* dface5 = new Triangle(dvert1, dvert2, dvert6, material);
		Triangle* dface6 = new Triangle(dvert2, dvert3, dvert6, material);
		Triangle* dface7 = new Triangle(dvert3, dvert4, dvert6, material);
		Triangle* dface8 = new Triangle(dvert4, dvert1, dvert6, material);
		objects.push_back(dface1);
		objects.push_back(dface2);
		objects.push_back(dface3);
		objects.push_back(dface4);
		objects.push_back(dface5);
		objects.push_back(dface6);
		objects.push_back(dface7);
		objects.push_back(dface8);
		*/

		std::vector<RectangleOwn> roomR = std::vector<RectangleOwn>{
			RectangleOwn(vec3(1, 0, 0), vec3(1, 1, 0), vec3(1, 1, 1), vec3(1, 0, 1), material),
			RectangleOwn(vec3(0, 0, 0), vec3(0, 1, 0), vec3(0, 1, 1), vec3(0, 0, 1), material),
			RectangleOwn(vec3(0, 1, 0), vec3(0, 1, 1), vec3(1, 1, 1), vec3(1, 1, 0), material),
			RectangleOwn(vec3(0, 0, 0), vec3(0, 0, 1), vec3(1, 0, 1), vec3(1, 0, 0), material),
			RectangleOwn(vec3(0, 0, 1), vec3(0, 1, 1), vec3(1, 1, 1), vec3(1, 0, 1), material),
			RectangleOwn(vec3(0, 0, 0), vec3(0, 1, 0), vec3(1, 1, 0), vec3(1, 0, 0), material)
		};
		Room* room = new Room(roomR);
		//objects.push_back(room);
		for (int i = 0; i < triangles.size(); i++)
			objects.push_back(triangles[i]);
		
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

	vec3 trace(Ray ray) {
		Hit hit = firstIntersect(ray);
		if (hit.t < 0) return vec3(0,0,0);
		vec3 outRadiance; 
		for (Light* light : lights) {
			float cosThetaIn = dot(ray.dir, hit.normal);
			float cosThetaOut = dot(-ray.dir, hit.normal);
			outRadiance = ((hit.material->ka / 2)*(1+ cosThetaOut)) * La;
			float L =  0.2 * (1 + cosThetaIn);
			if (L >= 2 && L <= 4 ) outRadiance =+ L;
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
	//scene.Animate(0.01f);
	//scene.build();

	long time = glutGet(GLUT_ELAPSED_TIME); // elapsed time since the start of the program
}
