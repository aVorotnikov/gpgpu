#ifndef __WIN_H_
#define __WIN_H_

#pragma comment(lib, "LIB/freeglut.lib")
#include "INCLUDE/glatter/glatter.h"
#include "INCLUDE/GL/freeglut.h"
#include "time.h"

#include "particle.cuh"


struct cudaGraphicsResource;

// класс win используется в случае выигрыша
class win
{
private:
  int W, H;
  UINT screenBuf; 
  UINT fboId = 0; 
  cudaGraphicsResource* screenRes; // представление ресурсов cuda
  part_mgr partMgr;
  int shapeSelected = -1;
  int prevX = 0, prevY = 0;


public:
  static win Instance;

  static void Display(void);

  static void Keyboard(unsigned char Key, int x, int y);

  static void Run(void);

  win(void);

  ~win(void);
};

#endif // __WIN_H_
