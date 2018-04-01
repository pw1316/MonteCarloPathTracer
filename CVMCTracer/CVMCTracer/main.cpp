#include "Framework/Math.hpp"
#include "Framework/ObjReader.hpp"
#include "CUDA/CUTracer.h"
#include <opencv/cv.hpp>

#define RENDER_1
//#define RENDER_2

int main()
{
    PW::FileReader::ObjModel model;
    PWVector3f *hostcolor = new PWVector3f[IMG_HEIGHT * IMG_WIDTH]; // Width*Height
    IplImage *img = nullptr;

#ifdef RENDER_1
    model.readObj("Resources/scene01.obj");
    PW::Tracer::CreateGeometry(&model);
    PW::Tracer::RenderScene(1, hostcolor);
    img = cvCreateImage(cvSize(IMG_WIDTH, IMG_HEIGHT), IPL_DEPTH_8U, 3);
    for (int i = 0; i < IMG_WIDTH; i++)
    {
        for (int j = 0; j < IMG_HEIGHT; j++)
        {
            PWVector3f &color = hostcolor[j * IMG_WIDTH + i];
            cvSet2D(img, j, i, CvScalar(color.z * 255, color.y * 255, color.x * 255));
        }
    }
    cvSaveImage("result1.png", img);
    cvReleaseImage(&img);
#endif // RENDER_1

#ifdef RENDER_2
    model.readObj("Resources/scene02.obj");
    PW::Tracer::CreateGeometry(&model);
    PW::Tracer::RenderScene(2, hostcolor);
    img = cvCreateImage(cvSize(IMG_WIDTH, IMG_HEIGHT), IPL_DEPTH_8U, 3);
    for (int i = 0; i < IMG_WIDTH; i++)
    {
        for (int j = 0; j < IMG_HEIGHT; j++)
        {
            PWVector3f &color = hostcolor[j * IMG_WIDTH + i];
            cvSet2D(img, j, i, CvScalar(color.z * 255, color.y * 255, color.x * 255));
        }
    }
    cvSaveImage("result2.png", img);
    cvReleaseImage(&img);
#endif // RENDER_2

    delete[] hostcolor;
}