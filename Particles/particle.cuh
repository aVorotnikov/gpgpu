#pragma once
#include "INCLUDE/glatter/glatter.h"
#include "INCLUDE/GL/freeglut.h"
#include "physics.cuh"

#include "cuda_runtime.h" 
#include "cuda_gl_interop.h" 

enum physics_type;

enum part_type
{
  PART_DEAD,
  PART_FIRST,
  PART_SECOND,
  PART_THIRD,
};

//структура частиц
struct particle
{
  float x, y;   // координаты
  float vx, vy; // скорость
  part_type type; //тип
  float remainingAliveTime; // оставшееся время жизни
  float originAliveTime;	// начальное время жизни
  physics_type phType; // физический тип

};

// описание данных генератора частицы
struct spawner
{
  float x, y; // координаты
  float vx, vy; // скорость
  part_type type; //тип частицы
  float spread; //распространение
  float intensity; //интенсивность
  int directionsCount; //направление
  float particleAliveTime; //время жизни
  physics_type phType; // физический тип
};

struct spawner_cbuf
{
  int nSpawners; 
  spawner spawners[20];
};

// тип фигуры
enum shape_type
{
  SHAPE_SQUARE,
  SHAPE_SEGMENT
};

// координаты фигуры: массив из 4 элементов
struct shape
{
  shape_type type;
  float params[4]; // x1 y1 x2 y2 для квадрата и отрезка, (x1>x2, y1>y2)
                   //cx, cy, radius, 0 для круга
};

// массив для константного буфера? 
struct shapes_cbuf
{
  int nShapes;
  shape shapes[20 + 3]; // последние 3 элемента для корзины
};

// координаты корзины
struct basket
{
  int x1, y1, x2, y2;
};

// определение констант и методов
class part_mgr
{
  particle* partPoolCur;
  spawner_cbuf spawnersHost;
  shapes_cbuf shapesHost;
  basket basketHost;
  int numInbasket;

public:
  static const int MAX_PARTICLES = 2000;
  static const int MAX_SPAWNERS = 20;
  static const int MAX_SHAPES = 20;
  static const int NUM_INBASKET_PARTS_TO_WIN = 100;

  void Init(void);
  void Compute(cudaSurfaceObject_t s, dim3 texSize, double timeDelta);
  void Kill(void);
  void AddSquare(float x1, float y1, float x2, float y2);
  void AddSegment(float x1, float y1, float x2, float y2);
  void MoveLastFigure(float dx, float dy);
  const shapes_cbuf& GetShapes(void);
  int SelectShape(int x, int y); 
  void MoveShape(int shapeHandle, int dx, int dy);
  int GetInbasket(void) { return numInbasket; }
};

// объявление константных переменных
__device__ __constant__ spawner_cbuf spawnersDevice;
__device__ __constant__ shapes_cbuf shapesDevice;
__device__ __constant__ basket basketDevice;
__device__ int inbasketParticlesCount; // количество частиц в корзине

//объявление функций
__global__ void Fill(cudaSurfaceObject_t s, dim3 texDim);
__device__ void SquareCollision(shape* shape, particle* part, float shiftX, float shiftY);
__device__ void SegmentCollision(shape* shape, particle* part, float shiftX, float shiftY);
__device__ void ShapesCollisionCheck(particle* part, double timeDelta);
