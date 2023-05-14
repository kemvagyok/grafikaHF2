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

struct plane {
	std::vector<vec3> points;
	vec3 normalVector;
	plane( std::vector<vec3> points0)
	{
		points = points0;
		normalVector = cross(points[1] - points[0], points[2] - points[0]);
	}

	bool isInsideArea(vec3 foundPoint)
	{
		for (int i = 0; i < points.size(); i++)
		{
			vec3 crossActual = cross(points[i] - points[(i == points.size() - 1) ? 0 : i + 1], foundPoint - points[(i == points.size() - 1) ? 0 : i + 1]);
			float dotActual = dot(crossActual, normalVector*-1);
			if (dotActual <= 0) return false;
		}
		return true;
	}

	void transform(float s, float d)
	{
		for (int i = 0; i < points.size(); i++)
			points[i] = points[i] * s + d;
	}
};


struct PlatonBody : Intersectable
{
	std::vector<vec3> points{
		vec3(0,0,0),
		vec3(1,0,0),
		vec3(0,1,0),
		vec3(0,0,1)
	};
	std::vector<plane> planes = std::vector<plane>{
		plane(std::vector<vec3>{points[0],points[1],points[2]}),
		plane(std::vector<vec3>{points[0],points[3],points[2]}),
		plane(std::vector<vec3>{points[0],points[1],points[3]}),
		plane(std::vector<vec3>{points[1],points[2],points[3]})
	};
	PlatonBody(Material* material0)
	{
		material = material0;

	}
	/*
	std::vector<vec3> points{
		vec3(0,-0.525731,0.850651),
		vec3(0.850651,0,0.525731),
		vec3(0.850651,0,-0.525731),
		vec3(-0.850651,0,-0.525731),
		vec3(-0.850651,0,0.525731),
		vec3(-0.525731,0.850651,0),
		vec3(0.525731,0.850651,0),
		vec3(0.525731,-0.850651,0),

		vec3(-0.525731,-0.850651,0),
		vec3(0,-0.525731,-0.850651),
		vec3(0,0.525731,-0.850651),
		vec3(0,0.525731 ,0.850651)
	};
	std::vector<plane> planes = std::vector<plane>{
		plane(std::vector<vec3>{points[1],points[2],points[6]}),
		plane(std::vector<vec3>{points[1],points[7],points[2]}),
		plane(std::vector<vec3>{points[3],points[4],points[5]}),
		plane(std::vector<vec3>{points[4],points[3],points[8]}),
		plane(std::vector<vec3>{points[6],points[5],points[11]}),
		plane(std::vector<vec3>{(points[5]),(points[6]),(points[10])}),
		plane(std::vector<vec3>{(points[9]),(points[10]),(points[2])}),
		plane(std::vector<vec3>{(points[10]),(points[9]),(points[3])}),
		plane(std::vector<vec3>{(points[7]),(points[8]),(points[9])}),
		plane(std::vector<vec3>{(points[8]),(points[7]),(points[0])}),
		plane(std::vector<vec3>{(points[11]),(points[0]),(points[1])}),
		plane(std::vector<vec3>{(points[0]),(points[11]),(points[4])}),
		plane(std::vector<vec3>{(points[6]),(points[2]),(points[10])}),
		plane(std::vector<vec3>{(points[1]),(points[6]),(points[11])}),
		plane(std::vector<vec3>{(points[3]),(points[5]),(points[10])}),
		plane(std::vector<vec3>{(points[5]),(points[4]),(points[11])}),
		plane(std::vector<vec3>{(points[2]),(points[7]),(points[9])}),
		plane(std::vector<vec3>{(points[7]),(points[1]),(points[0])}),
		plane(std::vector<vec3>{(points[3]),(points[9]),(points[8])}),
		plane(std::vector<vec3>{(points[4]),(points[8]),(points[0])})
	};
	*/
	/*
	std::vector<vec3> points{
	vec3(1,0,0),
	vec3(0,-1,0),
	vec3(-1,0,0),
	vec3(0,1,0),
	vec3(0,0,1),
	vec3(0,0,-1)
	};
	std::vector<plane> planes = std::vector<plane>{
		plane(std::vector<vec3>{points[1],points[0],points[4]}),
		plane(std::vector<vec3>{points[2],points[1],points[4]}),
		plane(std::vector<vec3>{points[3],points[2],points[4]}),
		plane(std::vector<vec3>{points[0],points[3],points[4]}),
		plane(std::vector<vec3>{points[0],points[1],points[5]}),
		plane(std::vector<vec3>{(points[1]),(points[2]),(points[5])}),
		plane(std::vector<vec3>{(points[2]),(points[3]),(points[5])}),
		plane(std::vector<vec3>{(points[3]),(points[0]),(points[5])})
	};*/
	

	Hit intersect(const Ray& ray)
	{
		Hit hit;
		vec3 normal(0,0,0);
		float temptT = -INFINITY;
		//printf("UJABB SUGÁR\n");
		for (int i = 0; i < planes.size(); i++)
		{
			//if (dot(planes[i].normalVector, ray.dir) < 0) planes[i].normalVector =planes[i].normalVector * 1;
			float t = dot(planes[i].points[0] - ray.start, planes[i].normalVector) / dot(ray.dir, planes[i].normalVector);
			if (t > 0) {
				vec3 rayt = ray.start + ray.dir * t;
				vec3 point = rayt - planes[i].points[0];

				if (dot(point, planes[i].normalVector) < 0.1 && dot(point, planes[i].normalVector) > -0.1)
				{
					if (planes[i].isInsideArea(point))
					{

						/*
						printf("normalvektor -> x: %f y: %f z: %f\n", planes[i].normalVector.x, planes[i].normalVector.y, planes[i].normalVector.z);
						printf("ray.dir -> x: %f y: %f z: %f\n", ray.dir.x, ray.dir.y, ray.dir.z);
						printf("rayt -> x: %f y: %f z: %f\n", rayt.x, rayt.y, rayt.z);
						printf("point -> x: %f y: %f z: %f\n\n\n", point.x, point.y, point.z);
						*/
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
		vec3 asd = ray.start + ray.dir * temptT;
		if (temptT < 0) return hit;
		hit.material = material;
		hit.normal = normal;
		hit.t = temptT;
		hit.position = ray.start + ray.dir * hit.t;
		return hit;
	}

};

struct Room : public Intersectable {
	std::vector<plane> planes = std::vector<plane>
	{
		plane(std::vector<vec3>{vec3(1, 0, 0), vec3(1, 1, 0), vec3(1, 1, 1), vec3(1, 0, 1)}),
		plane(std::vector<vec3>{vec3(0, 0, 0), vec3(0, 1, 0), vec3(0, 1, 1), vec3(0, 0, 1)}),
		plane(std::vector<vec3>{vec3(0, 1, 0), vec3(0, 1, 1), vec3(1, 1, 1), vec3(1, 1, 0)}),
		plane(std::vector<vec3>{vec3(0, 0, 0), vec3(0, 0, 1), vec3(1, 0, 1), vec3(1, 0, 0)}),
		plane(std::vector<vec3>{vec3(0, 0, 1), vec3(0, 1, 1), vec3(1, 1, 1), vec3(1, 0, 1)}),
		plane(std::vector<vec3>{vec3(0, 0, 0), vec3(0, 1, 0), vec3(1, 1, 0), vec3(1, 0, 0)})
	};

	Room(Material* material0)
	{			
		material = material0;
	}

	void rectangleownTransform(float s, float d)
	{
		for (int i = 0; i < planes.size(); i++)
		{
			planes[i].transform(s, d);
		}
	}

	Hit intersect(const Ray& ray)
	{
		Hit hit;
		vec3 normal(1, 2, 3);
		float temptT = -INFINITY;
		for (int i = 0; i < planes.size(); i++)
		{
			float t = dot(planes[i].points[0] - ray.start, planes[i].normalVector) / dot(ray.dir, planes[i].normalVector);
			if (t > 0) {
				vec3 rayt = ray.start + ray.dir * t;
				vec3 point = rayt - planes[i].points[0];
				if (dot(point, planes[i].normalVector) < 0.1 && dot(point, planes[i].normalVector) > - 0.1)
				{
					if (planes[i].isInsideArea(point))
					{ 
							if (temptT < t)
							{
								temptT = t;
								normal = planes[i].normalVector;
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
		Material* material = new Material(kd1, ks1,1);
		//Room* room = new Room(material);
		PlatonBody* platonbody1 = new PlatonBody(material);
		//objects.push_back(room);
		objects.push_back(platonbody1);
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
