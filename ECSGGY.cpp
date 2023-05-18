//=============================================================================================
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

    Triangle(vec3 _r1, vec3 _r2, vec3 _r3)
    {
        r1 = _r1;
        r2 = _r2;
        r3 = _r3;

        normalVector = cross(r2 - r1, r3 - r1);
        vec3 kd1(0.2, 0.2, 0.2), ks1(0.1, 0.1, 0.1); 
        material =new Material(kd1, ks1,0.1);
    }


    bool isInsideArea(vec3 p)
    {
        vec3 cross1 = cross(r2 - r1, p - r1);
        vec3 cross2 = cross(r3 - r2, p - r2);
        vec3 cross3 = cross(r1 - r3, p - r3);
        float dot1 = dot(cross1, normalVector);
        float dot2 = dot(cross2, normalVector);
        float dot3 = dot(cross3, normalVector);
        if (dot1 > 0 && dot2 > 0 && dot3 > 0) return true;
        return false;
    }

    void transform(float d, float s)
    {
        r1 = r1 * d + s;
        r2 = r2 * d + s;
        r3 = r3 * d + s;
    }

    Hit intersect(const Ray& ray)
    {
        Hit hit;
        float t = dot(r1 - ray.start, normalVector) / dot(ray.dir, normalVector);
        if (t > 0) 
        {
            vec3 point = ray.start + ray.dir * t;
            vec3 vector = point - r1;
            if (dot(vector, normalVector) < 0.1 && dot(vector,normalVector) > -0.1)
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
    RectangleOwn(vec3 _r1, vec3 _r2, vec3 _r3, vec3 _r4) : Triangle(_r1,_r2,_r3)
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
            vec3 point = ray.start + ray.dir * t;
            vec3 vector = point - r1 ;
            if (dot(vector, normalVector) < 0.1 && dot(vector, normalVector) > -0.1)
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

struct Cone : Intersectable
{
    vec3 p;
    vec3 normalVector;
    float alfa;
    float h;
    Cone(vec3 p0, float h0, float alfa0, vec3 normalVector0, Material* material0)
    {
        p = p0;        
        h = h0;
        alfa = alfa0;

        normalVector = normalize(normalVector0);
        vec3 kd1(0.2, 0.2, 0.2), ks1(0.1, 0.1, 0.1);
        material = material0;
    }

    Hit intersect(const Ray& ray)
    {
        Hit hit;

        float d2 = dot(ray.dir, ray.dir);
        float cosAlpha2 = pow(cosf(2*M_PI*(alfa/360)),1);
        vec3 dist = ray.start - p;

        float b = 2.0f * (dot(ray.dir , normalVector) * dot(dist , normalVector) - cosAlpha2 * dot(ray.dir , dist));
        float c = dot(dist , normalVector) * dot(dist , normalVector) - cosAlpha2 * dot(dist , dist);
        float a = pow(dot(ray.dir, normalVector), 2) - d2 * cosAlpha2;
        float discr = b * b - 4 * a * c;

        if (discr < 0) return hit; else discr = sqrtf(discr);
        float t1 = (-b + discr) / 2 / a;
        float t2 = (-b - discr) / 2 / a;
        vec3 r1 = ray.start + ray.dir * t1;
        vec3 r2 = ray.start + ray.dir * t2;
        float length1 = dot((r1 - p), normalVector);
        float length2 = dot((r2 - p), normalVector);
      
       if ((0 <= length1 && length1 <= h) && (0 <= length2 && length2 <= h))
       {
            float t; vec3 position;
            if (length2 >= length1) { t = t2; position = r2; }
            if (length2 <= length1) { t = t1; position = r1; }
            hit.t = t;
            hit.position = position;
            hit.normal = 2 * dot((position - p), normalVector) * normalVector - 2 * (position - p) * cosAlpha2;
            hit.material = material;
            return hit;
        }
        else if ((0 <= length1 && length1 <= h))
        {
            float t; vec3 position;
            t = t1; 
            position = r1; 
            hit.t = t;
            hit.position = position;
            hit.normal = 2 * dot((position - p), normalVector) * normalVector - 2 * (position - p) * cosAlpha2;
            hit.material = material;
            return hit;
        }
        else if ((0 <= length2 && length2 <= h))
        {
            float t; vec3 position;
            t = t2;
            position = r2;
            hit.t = t;
            hit.position = position;
            hit.normal = 2 * dot((position - p), normalVector) * normalVector - 2 * (position - p) * cosAlpha2;
            hit.material = material;
            return hit;
        }         
       return hit;
    }
    
    bool LightIntersect(const Hit& hit)
    {
        float hitCosAlfa = dot(normalize(hit.position - p), normalVector);
        if (hitCosAlfa >= cosf(2 * M_PI * (alfa / 360)))
            return true;
        
        return false;
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

class Scene {
    std::vector<Intersectable*> objects;
    std::vector<Light*> lights;
    std::vector<Cone*> cones;
    Camera camera;
    vec3 La;
public:
    void build() {


        float dt = M_PI*0;
        vec3 eye = vec3(2, 0.5, 2);
        vec3 vup = vec3(0, 1, 0);
        vec3 lookat = vec3(0, 0.75, 0);
        eye = vec3((eye.x - lookat.x) * cos(dt) + (eye.z - lookat.z) * sin(dt) + lookat.x,
            eye.y,
            -(eye.x - lookat.x) * sin(dt) + (eye.z - lookat.z) * cos(dt) + lookat.z);
        float fov = 45 * M_PI / 180;
        camera.set(eye, lookat, vup, fov);

        La = vec3(1, 1, 1);
        

        vec3 lightDirection1(1, -1, 1), Le1(0.8, 0, 0);
        lights.push_back(new Light(lightDirection1, Le1));
        vec3 lightDirection2(1, 0, 0), Le2(0, 0.5, 0);
        lights.push_back(new Light(lightDirection2, Le2));
        vec3 lightDirection3(0, 0, 1), Le3(0, 0, 0.5);
        lights.push_back(new Light(lightDirection3, Le3));

        
        std::vector<vec3> points1{
        vec3(1,0,0),
        vec3(0,-1,0),
        vec3(-1,0,0),
        vec3(0,1,0),
        vec3(0,0,1),
        vec3(0,0,-1)
        };
        for (int i = 0; i < points1.size(); i++)
            points1[i] = points1[i] * 0.25 + vec3(0.85, 0.25, 0.25);
            
        std::vector<Triangle*> platon1 = std::vector<Triangle*>{
          new Triangle(points1[1],points1[0],points1[4]),
          new Triangle(points1[2],points1[1],points1[4]),
          new Triangle(points1[3],points1[2],points1[4]),
          new Triangle(points1[0],points1[3],points1[4]),
          new Triangle(points1[0],points1[1],points1[5]),
          new Triangle(points1[1],points1[2],points1[5]),
          new Triangle(points1[2],points1[3],points1[5]),
          new Triangle(points1[3],points1[0],points1[5])
        };
       

        for (int i = 0; i < platon1.size(); i++)
            objects.push_back(platon1[i]);

        std::vector<vec3> points2{
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

        for (int i = 0; i < points2.size(); i++)
            points2[i] = points2[i] * 0.2 + vec3(0.25, 0.20, 0.25);
        
        
        
        std::vector<Triangle*> platon2 = std::vector<Triangle*>{
      new Triangle(points2[1],points2[2],points2[6]),
      new Triangle(points2[1],points2[7],points2[2]),
      new Triangle(points2[3],points2[4],points2[5]),
      new Triangle(points2[4],points2[3],points2[8]),
      new Triangle(points2[6],points2[5],points2[11]),
      new Triangle(points2[5],(points2[6]),points2[10]),
      new Triangle(points2[9],(points2[10]),points2[2]),
      new Triangle(points2[10],(points2[9]),points2[3]),
      new Triangle(points2[7],(points2[8]),points2[9]),
      new Triangle(points2[8],(points2[7]),points2[0]),
      new Triangle(points2[11],(points2[0]),points2[1]),
      new Triangle(points2[0],(points2[11]),points2[4]),
      new Triangle(points2[6],(points2[2]),points2[10]),
      new Triangle(points2[1],(points2[6]),points2[11]),
      new Triangle(points2[3],(points2[5]),points2[10]),
      new Triangle(points2[5],(points2[4]),points2[11]),
      new Triangle(points2[2],(points2[7]),points2[9]),
      new Triangle(points2[7],(points2[1]),points2[0]),
      new Triangle(points2[3],(points2[9]),points2[8]),
      new Triangle(points2[4],(points2[8]),points2[0])
        };
        for (int i = 0; i < platon2.size(); i++)
           objects.push_back(platon2[i]);
        Material* m1 = new Material(vec3(0.5, 0.2, 0.2), vec3(0.1, 0.1, 0.1),0.1);
        Material* m2 = new Material(vec3(0.2, 0.5, 0.2), vec3(0.1, 0.1, 0.1),0.1);
        Material* m3 = new Material(vec3(0.2, 0.2, 0.5), vec3(0.1, 0.1, 0.1),0.1);
        Cone* cone1 = new Cone(vec3(0, 1, 0), 0.25 , 20, vec3(1, -2, 1), m1);
        objects.push_back(cone1);
        Cone* cone2 = new Cone(vec3(0, 0.75, 0.75), 0.25 , 20, vec3(1, -1.5, -1.5), m2);
        objects.push_back(cone2);
        Cone* cone3 = new Cone(vec3(0.5, 0.75, 0), 0.25 , 20, vec3(0.5, -1.5, 1),m3);
        objects.push_back(cone3);
        cones =  std::vector<Cone*>{cone1, cone2, cone3};
        std::vector<RectangleOwn> roomR = std::vector<RectangleOwn>{
           RectangleOwn(vec3(1, 0, 0), vec3(1, 1, 0), vec3(1, 1, 1), vec3(1, 0, 1)),
            RectangleOwn(vec3(0, 0, 0), vec3(0, 1, 0), vec3(0, 1, 1), vec3(0, 0, 1)),
            RectangleOwn(vec3(0, 1, 0), vec3(0, 1, 1), vec3(1, 1, 1), vec3(1, 1, 0)),
            RectangleOwn(vec3(0, 0, 0), vec3(0, 0, 1), vec3(1, 0, 1), vec3(1, 0, 0)),
            RectangleOwn(vec3(0, 0, 1), vec3(0, 1, 1), vec3(1, 1, 1), vec3(1, 0, 1)),
            RectangleOwn(vec3(0, 0, 0), vec3(0, 1, 0), vec3(1, 1, 0), vec3(1, 0, 0))
        };      
        Room* room = new Room(roomR);
        
        objects.push_back(room);        
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
            Hit hit = object->intersect(ray); 
            if (hit.t > 0 && (bestHit.t < 0 || hit.t < bestHit.t))  bestHit = hit;
        }
        if (dot(ray.dir, bestHit.normal) > 0) bestHit.normal = bestHit.normal * (-1);
        return bestHit;
    }
    
    bool shadowIntersect(Ray ray) {	
        for (Intersectable* object : objects) if (object->intersect(ray).t > 0) return true;
        return false;
    }



    vec3 trace(Ray ray) {
        Hit hit = firstIntersect(ray);
        if (hit.t < 0) return vec3(0, 0, 0);
        vec3 outRadiance;
        float cosThetaIn = dot(normalize(ray.dir), normalize(hit.normal));
        float cosThetaOut = dot(normalize(ray.start - hit.position), normalize(hit.normal));
        outRadiance = outRadiance + ((hit.material->ka / 2) * (1 + cosThetaOut)) * La;
        float L = 0.2 * (1 + cosThetaIn);
        if (L >= 0.2 && L <= 0.4) outRadiance = outRadiance + L;

        for (int i = 0; i < cones.size(); i+=1 ) {
            if (cones[i]->LightIntersect(hit))
            {
                Ray ray = Ray(cones[i]->p, normalize(hit.position - cones[i]->p));
                Hit hitIntersect = firstIntersect(ray);
                if (hitIntersect.position.x > hit.position.x - 0.1 && hitIntersect.position.x < hit.position.x +  0.1 
                    && hitIntersect.position.y > hit.position.y - 0.1 && hitIntersect.position.y < hit.position.y + 0.1  
                    && hitIntersect.position.z > hit.position.z - .1 && hitIntersect.position.z < hit.position.z + 0.1)
                {
                    outRadiance = outRadiance + (((hit.material->ka / 2) * (1 + cosThetaOut)) * lights[i]->Le) * (1 / length(hit.position - cones[i]->p));
                }
            }
        }
        return outRadiance;
    }
};


GPUProgram gpuProgram;
Scene scene;
unsigned int vao;	 

class FullScreenTexturedQuad {
    unsigned int vao;
    Texture texture;
public:
    FullScreenTexturedQuad(int windowWidth, int windowHeight, std::vector<vec4>& image)
        : texture(windowWidth, windowHeight, image)
    {
        glGenVertexArrays(1, &vao);	
        glBindVertexArray(vao);	

        unsigned int vbo;		
        glGenBuffers(1, &vbo);	

     
        glBindBuffer(GL_ARRAY_BUFFER, vbo); 
        float vertexCoords[] = { -1, -1,  1, -1,  1, 1,  -1, 1 };	
        glBufferData(GL_ARRAY_BUFFER, sizeof(vertexCoords), vertexCoords, GL_STATIC_DRAW);	   
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);     
    }

    void Draw() {
        glBindVertexArray(vao);	
        gpuProgram.setUniform(texture, "textureUnit");
        glDrawArrays(GL_TRIANGLE_FAN, 0, 4);	
    }
};
FullScreenTexturedQuad* fullScreenTexturedQuad;

void onInitialization() {
    glViewport(0, 0, windowWidth, windowHeight);
    scene.build();

    std::vector<vec4> image(windowWidth * windowHeight);
    long timeStart = glutGet(GLUT_ELAPSED_TIME);
    scene.render(image);
    long timeEnd = glutGet(GLUT_ELAPSED_TIME);
    printf("Rendering time: %d milliseconds\n", (timeEnd - timeStart));

    fullScreenTexturedQuad = new FullScreenTexturedQuad(windowWidth, windowHeight, image);

    gpuProgram.create(vertexSource, fragmentSource, "fragmentColor");
}

void onDisplay() {
    fullScreenTexturedQuad->Draw();
    glutSwapBuffers();
}

void onKeyboard(unsigned char key, int pX, int pY) {
    if (key == 'd') glutPostRedisplay();         // if d, invalidate display, i.e. redraw
}

void onKeyboardUp(unsigned char key, int pX, int pY) {
}

void onMouseMotion(int pX, int pY) {	
    float cX = 2.0f * pX / windowWidth - 1;	
    float cY = 1.0f - 2.0f * pY / windowHeight;
    printf("Mouse moved to (%3.2f, %3.2f)\n", cX, cY);
}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) { 
    float cX = 2.0f * pX / windowWidth - 1;	
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

void onIdle() {
    long time = glutGet(GLUT_ELAPSED_TIME); 
}
