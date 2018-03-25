#include "Framework/Math.hpp"
#include "Framework/ObjReader.hpp"
#include "CUDA/CUTracer.h"
#include <opencv/cv.hpp>

//#define RENDER_1
#define RENDER_2

int main()
{
    PW::FileReader::ObjModel model;
    PWVector4f *hostcolor = new PWVector4f[IMG_HEIGHT * IMG_WIDTH]; // Width*Height

#ifdef RENDER_1
    model.readObj("Resources/scene01.obj");
    PW::Tracer::RenderScene(1, &model, hostcolor);
    IplImage *img = cvCreateImage(cvSize(IMG_WIDTH, IMG_HEIGHT), IPL_DEPTH_8U, 3);
    for (int i = 0; i < IMG_WIDTH; i++)
    {
        for (int j = 0; j < IMG_HEIGHT; j++)
        {
            PWVector4f &color = hostcolor[j * IMG_WIDTH + i];
            cvSet2D(img, j, i, CvScalar(color.z * 255, color.y * 255, color.x * 255));
        }
    }
    cvShowImage("result1", img);
    cvWaitKey(0);
    cvSaveImage("result1.png", img);
#endif // RENDER_1

#ifdef RENDER_2
    model.readObj("Resources/scene02.obj");
    PW::Tracer::RenderScene(2, &model, hostcolor);
    IplImage *img = cvCreateImage(cvSize(IMG_WIDTH, IMG_HEIGHT), IPL_DEPTH_8U, 3);
    for (int i = 0; i < IMG_WIDTH; i++)
    {
        for (int j = 0; j < IMG_HEIGHT; j++)
        {
            PWVector4f &color = hostcolor[j * IMG_WIDTH + i];
            cvSet2D(img, j, i, CvScalar(color.z * 255, color.y * 255, color.x * 255));
        }
    }
    cvShowImage("result2", img);
    cvWaitKey(0);
    cvSaveImage("result2.png", img);
#endif // RENDER_2

    delete[] hostcolor;
}