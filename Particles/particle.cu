#include "particle.cuh"
#include "physics.cuh"
#include "math.h"

#include "cudaGL.h"
#include "device_launch_parameters.h"

//
//  функции для частиц
//

// заполнить текстуру указанным цветом
__global__ void Fill(cudaSurfaceObject_t s, dim3 texDim)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= texDim.x || y >= texDim.y)
    {
        return;
    }

    uchar4 data = make_uchar4(0x8d, 0xc4, 0x99, 0xff); // цвет ткекстуры
    surf2Dwrite(data, s, x * sizeof(uchar4), y); //записывает значения в 2d массиы cuda
}

// отрисовка фигур 
// cudaSurfaceObject_t - объект-поверхность в cuda
// blockIdx - индекс блока 
__global__ void DrawShapes(cudaSurfaceObject_t s)
{
    unsigned int i = blockIdx.x;
    if (i > shapesDevice.nShapes)
        return;
    shape shp = shapesDevice.shapes[i];
}

// отрисовка частиц
__global__ void DrawParticles(cudaSurfaceObject_t s, particle* poolCur, dim3 texDim)
{
    unsigned int i = blockIdx.x;
    particle part = poolCur[i];
    unsigned int x = part.x;
    unsigned int y = part.y;

    if (x + 1 >= texDim.x || y + 1 >= texDim.y || part.type == PART_DEAD)
        return;

    // раскрашиваем частицы разного типа в разные цвета
    float r = 0, g = 0, b = 0;
    switch (part.type)
    {
    case PART_FIRST: {r = 255; g = 255; break; }
    case PART_SECOND: {r = 128; g = 64; break; }
    case PART_THIRD: {g = 128; break; }
    }

    uchar4 data = make_uchar4(r, g, b, 0xff); //определяем цвет частички

    surf2Dwrite(data, s, x * sizeof(uchar4), y);
    surf2Dwrite(data, s, (x + 1) * sizeof(uchar4), y);
    surf2Dwrite(data, s, x * sizeof(uchar4), y + 1);
    surf2Dwrite(data, s, (x + 1) * sizeof(uchar4), y + 1);
}

// вычисление расстояния между частицами
__device__ float dist(particle x, particle y)
{
    return float(sqrt(pow((x.x - y.x), 2) + pow((x.y - y.y), 2)));
}

// проверка столкновения
__device__ void CollisionCheck(particle* poolCur, int maxParticles)
{
    unsigned int i = blockIdx.x;

    float t;
    // если частички еще живы, они разного типа и расстояние между центрами меньше 2, то они должны столкнуться и поменять направления скорости
    for (int j = i + 1; j < maxParticles; j++)
        if (poolCur[i].type != PART_DEAD && poolCur[j].type != PART_DEAD)
            if (poolCur[i].type != poolCur[j].type)
                if (dist(poolCur[i], poolCur[j]) < 2)
                {
                    t = poolCur[i].vx;
                    poolCur[i].vx = poolCur[j].vx;
                    poolCur[j].vx = t;

                    t = poolCur[i].vy;
                    poolCur[i].vy = poolCur[j].vy;
                    poolCur[j].vy = t;
                }
}

// обновление состояния частиц
__global__ void Update(particle* poolCur, double timeDelta, int maxParticles, dim3 texDim)
{
    unsigned int i = blockIdx.x;
    //phManager.physicsMakeAction(&poolCur[i]);
    // если частица вылетела за границы, то она исчезает
    if (poolCur[i].x + 1 >= texDim.x || poolCur[i].y + 1 >= texDim.y)
        poolCur[i].type = PART_DEAD;
    if (poolCur[i].type == PART_DEAD)
        return;

    //проверяем на столкновение с предметом (фигурой)
    ShapesCollisionCheck(&poolCur[i], timeDelta);
    poolCur[i].vy -= 0.00015 * timeDelta;   
    poolCur[i].x = poolCur[i].x + poolCur[i].vx * timeDelta;
    poolCur[i].y = poolCur[i].y + poolCur[i].vy * timeDelta;

    // проверяем столкновение частичек
    CollisionCheck(poolCur, maxParticles);
    // проверяем время жизни частички
    poolCur[i].remainingAliveTime = max(poolCur[i].remainingAliveTime - timeDelta, 0.f);
    // если частичка еще жива, то ок, если нет, то она исчезает
    if (poolCur[i].remainingAliveTime > 0) {
        poolCur[i].type = poolCur[i].type;
    }
    else {
        poolCur[i].type = PART_DEAD;
    }
}

// булево значение: попала ли частичка в корзину
__device__ bool btwDev(int x, int x1, int x2)
{
     return x >= x1 && x <= x2 || x >= x2 && x <= x1;
}

// проверка попали ли частицы в корзину
__global__ void CheckBasket(particle* poolCur, int maxParticles)
{
    // используем разделяемую память
    __shared__ int locNum;
    // unsigned int используется для неотрицательных значений (вычисляем индекс текущено потока)
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    // в первом потоке задаем значение количества элементов в корзине нулевым
    if (threadIdx.x == 0) {
        locNum = 0;
    }
    // __syncthreads(); - синхронизаци потоков (потоки приостанавливаются 
    // до того момента, пока все потоки не достигнут этой точки)
    __syncthreads();

    if (x >= maxParticles) {
        return;
    }
    particle p = poolCur[x];
    // проверяем тип частицы в потоке
    if (p.type == PART_DEAD) {
        return;
    }
    // сравниваем координаты частички и кооржинаты корзины
    if (btwDev(p.x, basketDevice.x1, basketDevice.x2) && btwDev(p.y, basketDevice.y1, basketDevice.y2)) {
        // увеличиваем на 1 количество частиц, попавших в корзину
        atomicAdd(&locNum, 1);
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        atomicAdd(&inbasketParticlesCount, locNum);
    }
}

// 
__device__ unsigned seed = 123456789;
__device__ unsigned random(void)
{
    unsigned a = 1103515245;
    unsigned c = 12345;
    unsigned m = 1 << 31;
    seed = (a * seed + c) % m;
    return seed;
}

// создание новых частиц
__global__ void Spawn(particle* poolCur, int maxParticles)
{
    int startSlot = 0;

    for (int i = 0; i < spawnersDevice.nSpawners; i++)
    {
        spawner sp = spawnersDevice.spawners[i];
        int numToSpawn = sp.intensity;
        for (int j = 0; j < numToSpawn; j++)
            for (int k = startSlot; k < maxParticles; k++)
                if (poolCur[k].type == PART_DEAD)
                {

                    particle p = { sp.x, sp.y, sp.vx + (random() % sp.directionsCount) * sp.spread, sp.vy + (random() % sp.directionsCount) * sp.spread,
                                      sp.type, sp.particleAliveTime, sp.particleAliveTime, sp.phType };

                    poolCur[k] = p;
                    startSlot = k + 1;
                    break;
                }
    }
}

//
// методы part_mgr - жизненный цикл системы
//

// выполнение расчетов и обновление
void part_mgr::Compute(cudaSurfaceObject_t s, dim3 texSize, double timeDelta)
{
    // cudaMemcpyToSymbol - копирование данных в константную память
    cudaMemcpyToSymbol(shapesDevice, &shapesHost, sizeof(shapes_cbuf));
    numInbasket = 0;
    cudaMemcpyToSymbol(inbasketParticlesCount, &numInbasket, sizeof(int));

    dim3 thread(1);
    dim3 block(MAX_PARTICLES);
    dim3 oneBlock(1);

    // вызываем функции создания частиц, обновления состояния, проверка корзины
    // отрисовка частиц
    Spawn << < oneBlock, thread >> > (partPoolCur, MAX_PARTICLES);
    Update << < block, thread >> > (partPoolCur, timeDelta, MAX_PARTICLES, texSize);
    CheckBasket << <dim3(MAX_PARTICLES / 32 + 1), dim3(32) >> > (partPoolCur, MAX_PARTICLES);
    DrawParticles << < block, thread >> > (s, partPoolCur, texSize);
    // копирование данных с устройства на хост
    cudaMemcpyFromSymbol(&numInbasket, inbasketParticlesCount, sizeof(int));
}

// инициализация создания частиц
void part_mgr::Init(void)
{
    // cudaStatus - для отслеживания и передачи информации 
    //об успешном или неуспешном выполнении операций
    cudaError_t cudaStatus = cudaSuccess;
    particle tmp[MAX_PARTICLES];
    for (int i = 0; i < MAX_PARTICLES; i++)
    {
        particle p = { 0, 0, 0, 0, PART_DEAD, 0, 0, SPACE };
        tmp[i] = p;
    }
    // cudaMalloc - выделение памяти на GPU
    cudaStatus = cudaMalloc(&partPoolCur, sizeof(particle) * MAX_PARTICLES);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "failed!");
    }

    // копирование данных с CPU на GPU
    cudaMemcpy(partPoolCur, tmp, sizeof(particle) * MAX_PARTICLES, cudaMemcpyHostToDevice);

    // создание трех объектов для генерации частичек трех типов
    spawnersHost.nSpawners = 3;
    spawnersHost.spawners[0] = { 200, 200, 0.30, 0.55, PART_FIRST, 0.005, 2, 8, 3000, EARTH_PHYSICS };
    spawnersHost.spawners[1] = { 850, 200, 0.15, 0.65, PART_SECOND, -0.008, 3, 10, 3000, SPACE };
    spawnersHost.spawners[2] = { 1000, 300, -0.35, 0.35, PART_THIRD, -0.005, 1, 10, 1500, SPACE };

    cudaMemcpyToSymbol(spawnersDevice, &spawnersHost, sizeof(spawner_cbuf));

    // координаты корзины
    basketHost = { 400, 5, 800, 150 };
    cudaMemcpyToSymbol(basketDevice, &basketHost, sizeof(basket));

    float x1 = basketHost.x1, x2 = basketHost.x2, y1 = basketHost.y1, y2 = basketHost.y2;

    // создание 3 отрезков (отрисовка корзины)
    shapesHost.shapes[MAX_SHAPES] = { SHAPE_SEGMENT, x1, y2, x1, y1 };
    shapesHost.shapes[MAX_SHAPES + 1] = { SHAPE_SEGMENT, x2, y1, x1, y1 };
    shapesHost.shapes[MAX_SHAPES + 2] = { SHAPE_SEGMENT, x2, y2, x2, y1 };

}

// завершить создание частиц, освободить память
void part_mgr::Kill(void)
{
    cudaError_t cudaStatus = cudaSuccess;
    // cudaFree - cudaFree
    cudaStatus = cudaFree(partPoolCur);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "failed!");
    }
}

//
// обработка предметов (фигур)
//

// добаывить квадрат
void part_mgr::AddSquare(float x1, float y1, float x2, float y2)
{
    if (shapesHost.nShapes == MAX_SHAPES) {
        return;
    }
    shapesHost.shapes[shapesHost.nShapes] = { SHAPE_SQUARE, x1, y1, x2, y2 };
    shapesHost.nShapes++;
}

// добавить отрезок
void part_mgr::AddSegment(float x1, float y1, float x2, float y2)
{
    if (shapesHost.nShapes == MAX_SHAPES) {
        return;
    }
    shapesHost.shapes[shapesHost.nShapes] = { SHAPE_SEGMENT, x1, y1, x2, y2 };
    shapesHost.nShapes++;
}

void part_mgr::MoveLastFigure(float dx, float dy)
{
     if (0 == shapesHost.nShapes) {
          return;
     }
     auto shape = shapesHost.shapes[shapesHost.nShapes - 1];
     shape.params[0] += dx;
     shape.params[2] += dx;
     shape.params[1] += dy;
     shape.params[3] += dy;
     shapesHost.shapes[shapesHost.nShapes - 1] = shape;
}

// создание фигур
const shapes_cbuf& part_mgr::GetShapes(void)
{
    return shapesHost;
}

// вычисление квадрата числа
inline int sqr(int x)
{
    return x * x;
}

// вычисление расстояния между тремя точками?
inline bool btw(int x, int x1, int x2)
{
    return x1 < x2 ? (x >= x1 && x <= x2) : (x <= x1 && x >= x2);
}

// вычисление расстояния
inline float dist(int x0, int y0, int x1, int y1, int x2, int y2)
{
    return (float)abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1) / sqrt(sqr(y2 - y1) + sqr(x2 - x1));
}

//поместить предмет (фигуру) в координаты
int part_mgr::SelectShape(int x, int y)
{
    for (int i = 0; i < shapesHost.nShapes; i++)
    {
        shape& shp = shapesHost.shapes[i];
        switch (shp.type)
        {
        case SHAPE_SQUARE:
            if (btw(x, shp.params[0], shp.params[2]) && btw(y, shp.params[1], shp.params[3])) {
                return i;
            }
        case SHAPE_SEGMENT:
            if (btw(x, shp.params[0], shp.params[2]) && btw(y, shp.params[1], shp.params[3]) &&
                dist(x, y, shp.params[0], shp.params[1], shp.params[2], shp.params[3]) < 5) {
                return i;
            }
            break;
        default:
            break;
        }
    }
    return -1;
}

// переместить предмет (фигуру)
// shapeHandle - объект,  которым можно управлять для изменения координат/формы/размера
void part_mgr::MoveShape(int shapeHandle, int dx, int dy)
{
    if (shapeHandle < 0 || shapeHandle >= shapesHost.nShapes)
        return;

    // получить координаты предмета(фигуры)
    shape& shp = shapesHost.shapes[shapeHandle];
    switch (shp.type)
    {
    case SHAPE_SQUARE:
    case SHAPE_SEGMENT:
        shp.params[0] += dx;
        shp.params[1] += dy;
        shp.params[2] += dx;
        shp.params[3] += dy;
        break;
    default:
        break;
    }
}

//
// обработка столкновений
//

// координаты частицы
struct pt
{
    float x, y;
};

// столкновение фигур и частиц
__device__ void ShapesCollisionCheck(particle* part, double timeDelta)
{
    float shiftX, shiftY;
    // умножаем скорость на время и получаем расстояние
    shiftX = part->vx * timeDelta;
    shiftY = part->vy * timeDelta;
    shape sh;

    // определяем тип предмета с которым сталкиваемся
    for (int i = 0; i < shapesDevice.nShapes; i++)
    {
        sh = shapesDevice.shapes[i];
        switch (sh.type)
        {
        case SHAPE_SQUARE: {SquareCollision(&sh, part, shiftX, shiftY); break; }
        case SHAPE_SEGMENT: {SegmentCollision(&sh, part, shiftX, shiftY); break; }
        //case SHAPE_CIRCLE: {CircleCollision(&sh, part, shiftX, shiftY); break; }
        }
    }

    // проверка попадания в корзину
    for (int i = 20; i < 23; i++)
    {
        sh = shapesDevice.shapes[i];
        switch (sh.type)
        {
        case SHAPE_SEGMENT: {SegmentCollision(&sh, part, shiftX, shiftY); break; }
        }
    }
}

// столкновение с квадратом
__device__ void SquareCollision(shape* shape, particle* part, float shiftX, float shiftY)
{
    float newX = part->vx, newY = part->vy;
    if (part->x + shiftX <= shape->params[0] && part->x + shiftX >= shape->params[2]
        && part->y <= shape->params[1] && part->y >= shape->params[3])
    {
        newX *= -0.2;   //замедление после столкновения
    }

    if (part->y + shiftY <= shape->params[1] && part->y + shiftY >= shape->params[3]
        && part->x <= shape->params[0] && part->x >= shape->params[2])
    {
        newY *= -0.2;    //замедление после столкновения
    }

    part->vx = newX;
    part->vy = newY;
}

__device__  int area(pt a, pt b, pt c)
{
    return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
}

__device__ int sign(float x)
{
    float eps = 0.1;
    if (x > eps) {
        return 1;
    }
    else {
        if (x < eps) {
            return -1;
        }
        else {
             return 0;
        }
    }
}

// определение пересечения
__device__  bool intersect_1(float a, float b, float c, float d)
{
    float t;
    if (a > b)
    {
        t = a;
        a = b;
        b = t;
    }
    if (c > d)
    {
        t = c;
        c = d;
        d = t;
    }
    return max(a, c) <= min(b, d);
}


// пересечение
__device__ bool intersect(pt a, pt b, pt c, pt d)
{
    return intersect_1(a.x, b.x, c.x, d.x)
        && intersect_1(a.y, b.y, c.y, d.y)
        && sign(area(a, b, c)) * sign(area(a, b, d)) <= 0
        && sign(area(c, d, a)) * sign(area(c, d, b)) <= 0;
}

//столкновение с сегментом
__device__ void SegmentCollision(shape* shape, particle* part, float shiftX, float shiftY)
{
    if (intersect(pt{ shape->params[0], shape->params[1] }, pt{ shape->params[2], shape->params[3] },
        pt{ part->x, part->y }, pt{ part->x + shiftX, part->y + shiftY }))
    {
        pt prtcl = { part->vx, part->vy };
        pt norm = { shape->params[1] - shape->params[3], -shape->params[0] + shape->params[2] };

        float side = sign(area(pt{ shape->params[0], shape->params[1] },
            pt{ shape->params[2], shape->params[3] }, pt{ part->x, part->y }));
        norm = { norm.x * side, norm.y * side };

        float len = sqrt(pow(norm.x, 2) + pow(norm.y, 2));
        norm = { norm.x / len, norm.y / len };
        float t = 2 * (prtcl.x * norm.x + prtcl.y * norm.y);
        norm = { t * norm.x, t * norm.y };
        pt res = { prtcl.x - norm.x, prtcl.y - norm.y };

        //изменение скорости после столкновения
        part->vx = res.x * 0.8; 
        part->vy = res.y * 0.8;
    }
}




