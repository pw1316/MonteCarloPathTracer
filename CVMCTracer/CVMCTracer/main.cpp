#include "Framework/Math.hpp"
#include "Framework/ObjReader.hpp"
#include "CUDA/CUTracer.h"
#include <opencv/cv.hpp>

int main()
{
    PW::FileReader::ObjModel model;
    model.readObj("Resources/scene01.obj");
    PWVector4f hostcolor[IMG_HEIGHT][IMG_WIDTH]; // Width*Height
    PW::Tracer::RenderScene1(&model, &(hostcolor[0][0]));
    IplImage *img = cvCreateImage(cvSize(IMG_WIDTH, IMG_HEIGHT), IPL_DEPTH_8U, 3);
    for (int i = 0; i < IMG_WIDTH; i++)
    {
        for (int j = 0; j < IMG_HEIGHT; j++)
        {
            PWVector4f &color = hostcolor[j][i];
            cvSet2D(img, j, i, CvScalar(color.z * 255, color.y * 255, color.x * 255));
        }
    }
    cvShowImage("a", img);
    cvWaitKey(0);
    cvSaveImage("a.png", img);
}