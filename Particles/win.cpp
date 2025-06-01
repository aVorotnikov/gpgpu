#include "win.h"
win win::Instance;


win::win(void) : W(1280), H(736)
{
  char* v[1] = { 0 };
  int c = 1;

  //  бибилиотека GLUT для создания окна и отрисовки графики
  glutInit(&c, v);

  glutInitDisplayMode(GLUT_RGB);
  glutInitWindowPosition(10, 10);
  glutInitWindowSize(W, H);
  glutCreateWindow("Particle game");
  glutDisplayFunc(Display);
  glutKeyboardFunc(Keyboard);
  // glutMouseFunc(Mouse);
  // glutMotionFunc(MouseMotion);

  glGenTextures(1, &screenBuf);
  glBindTexture(GL_TEXTURE_2D, screenBuf);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, W, H, 0, GL_RGBA, GL_FLOAT, nullptr);

  glGenFramebuffers(1, &fboId);
  glBindFramebuffer(GL_READ_FRAMEBUFFER, fboId);
  glFramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
    GL_TEXTURE_2D, screenBuf, 0);
  glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);  // if not already bound

  auto e = cudaGLSetGLDevice(0);
  e = cudaGraphicsGLRegisterImage(&screenRes, screenBuf, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore);

  partMgr.Init();
}

// преобразование координат пикселей в нормализованные
inline float toNdc(int cPixel, int pixelsNum)
{
  return (float)cPixel / (float)pixelsNum * 2.0 - 1.0;
}

// отображение окна, отрисовка фигур, вычисления, обновление
void win::Display(void)
{
  static clock_t startTime = clock();
  clock_t endTime;
  glClearColor(1.0f * rand() / RAND_MAX, 1.0f * rand() / RAND_MAX, 1.0f * rand() / RAND_MAX, 1.0f * rand() / RAND_MAX);
  glClear(GL_COLOR_BUFFER_BIT);

  auto e = cudaGraphicsMapResources(1, &Instance.screenRes);
  cudaArray_t writeArray;
  e = cudaGraphicsSubResourceGetMappedArray(&writeArray, Instance.screenRes, 0, 0);
  cudaResourceDesc wdsc;
  wdsc.resType = cudaResourceTypeArray;
  wdsc.res.array.array = writeArray;
  cudaSurfaceObject_t writeSurface;
  e = cudaCreateSurfaceObject(&writeSurface, &wdsc);


  // очистка заднего фона, запуск функции Fill
  dim3 thread(32, 32);
  dim3 texDim(Instance.W, Instance.H);
  dim3 block(texDim.x / thread.x, texDim.y / thread.y);
  Fill << < block, thread >> > (writeSurface, texDim);

  // вычисления 
  endTime = clock();
  double delta = (endTime - startTime) * 1000 / CLOCKS_PER_SEC; // вычисление времени в милисекундах
  Instance.partMgr.Compute(writeSurface, texDim, delta);
  startTime = endTime;

  // освобождение ресурсов и синхронизация 
  e = cudaDestroySurfaceObject(writeSurface);
  e = cudaGraphicsUnmapResources(1, &Instance.screenRes);
  e = cudaStreamSynchronize(0);

  glBlitFramebuffer(0, 0, Instance.W, Instance.H, 0, 0, Instance.W, Instance.H,
    GL_COLOR_BUFFER_BIT, GL_NEAREST);


  // отрисовка предметов (фигур)
  const shapes_cbuf& shapes = Instance.partMgr.GetShapes();
  float ratio = (float)Instance.W / Instance.H;
  // int lineAmount = 100; //количество треугольников, используемых для рисования круга
  // GLfloat twicePi = 2 * 3.141592;
  for (int i = 0; i < shapes.nShapes; i++)
  {
    int x1 = shapes.shapes[i].params[0], y1 = shapes.shapes[i].params[1], x2 = shapes.shapes[i].params[2], y2 = shapes.shapes[i].params[3];
    switch (shapes.shapes[i].type)
    {
    case SHAPE_SQUARE:
      glBegin(GL_LINE_LOOP);
      glVertex2f(toNdc(x1, Instance.W), toNdc(y1, Instance.H));
      glVertex2f(toNdc(x1, Instance.W), toNdc(y2, Instance.H));
      glVertex2f(toNdc(x2, Instance.W), toNdc(y2, Instance.H));
      glVertex2f(toNdc(x2, Instance.W), toNdc(y1, Instance.H));
      glEnd();
      break;
    case SHAPE_SEGMENT:
      glBegin(GL_LINES);
      glVertex2f(toNdc(x1, Instance.W), toNdc(y1, Instance.H));
      glVertex2f(toNdc(x2, Instance.W), toNdc(y2, Instance.H));
      glEnd();
      break;
    default:
      break;
    }
  }

  for (int i = Instance.partMgr.MAX_SHAPES; i < Instance.partMgr.MAX_SHAPES + 3; i++)
  {
    int x1 = shapes.shapes[i].params[0], y1 = shapes.shapes[i].params[1], x2 = shapes.shapes[i].params[2], y2 = shapes.shapes[i].params[3];
    switch (shapes.shapes[i].type)
    {
    case SHAPE_SEGMENT:
      glBegin(GL_LINES);
      glVertex2f(toNdc(x1, Instance.W), toNdc(y1, Instance.H));
      glVertex2f(toNdc(x2, Instance.W), toNdc(y2, Instance.H));
      glEnd();
      break;
    default:
      break;
    }
  }
  char buf[200] = { 0 };
  sprintf(
       buf,
       "Press q and l for spawn rectangle and segment. Current parts in basket: %i / %i.",
       Instance.partMgr.GetInbasket(),
       Instance.partMgr.NUM_INBASKET_PARTS_TO_WIN);
  glutSetWindowTitle(buf);

  //если в корзину попало необходимое число частиц, появляется надпись  "YOU WIN"
  if (Instance.partMgr.GetInbasket() >= Instance.partMgr.NUM_INBASKET_PARTS_TO_WIN)
  {
    glRasterPos2f(0, 0);
    unsigned char bb[] = "YOU WIN";
    glutBitmapString(GLUT_BITMAP_HELVETICA_18, bb);
  }

  glFinish();
  glutSwapBuffers();
  glutPostRedisplay();
}

// обработка событий с клавиатуры для добавления предметов
void win::Keyboard(unsigned char Key, int x, int y)
{
  if (Key == 27)
    exit(0);
  if (Key == 'F' || Key == 'f')
    glutFullScreenToggle();
  if (Key == 'Q' || Key == 'q')
    Instance.partMgr.AddSquare(400, 650, 150, 500);
  if (Key == 'L' || Key == 'l')
    Instance.partMgr.AddSegment(650, 700, 1000, 500);

  float dx = 0, dy = 0;
  if (Key == 'W' || Key == 'w')
       dy += 50;
  if (Key == 'S' || Key == 's')
       dy -= 50;
  if (Key == 'A' || Key == 'a')
       dx -= 50;
  if (Key == 'D' || Key == 'd')
       dx += 50;
  Instance.partMgr.MoveLastFigure(dx, dy);
}

//запуск главного цикла обработки и отображения окна
void win::Run(void)
{
  glutMainLoop();
}

// освобождение ресурсов и завершение работы программы
win::~win(void)
{
  partMgr.Kill();

  cudaError_t cudaStatus = cudaSuccess;
  cudaStatus = cudaDeviceReset();
  if (cudaStatus != cudaSuccess)
    fprintf(stderr, "cudaDeviceReset failed!");
}

