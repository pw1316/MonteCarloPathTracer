#include "MCTracer.h"
#include <QtWidgets/QApplication>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    MCTracer w;
    w.show();
    return a.exec();
}
