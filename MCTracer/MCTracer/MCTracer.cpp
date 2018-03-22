#include "MCTracer.h"

#include <QtWidgets/QApplication>

#include "CUDA/CUTracer.h"
#include "Framework/ObjReader.hpp"

MCTracer::MCTracer(QWidget *parent) : QMainWindow(parent)
{
    InitUI();
    BindSignalSlot();
}

void MCTracer::InitUI()
{
    if (this->objectName().isEmpty())
    {
        this->setObjectName(QStringLiteral("MCTracer"));
    }
    this->setWindowTitle(QApplication::translate("MCTracer", "MCTracer", Q_NULLPTR));
    this->resize(1030, 768);
    this->setMinimumSize(QSize(1030, 768));
    this->setMaximumSize(QSize(1030, 768));

    menuBar = new QMenuBar(this);
    this->setMenuBar(menuBar);
    menuBar->setObjectName(QStringLiteral("menuBar"));
    menuBar->setGeometry(QRect(0, 0, 1030, 23));

    menuFile = new QMenu(menuBar);
    menuFile->setTitle(QApplication::translate("MCTracer", "&File", Q_NULLPTR));
    menuFile->setObjectName(QStringLiteral("menuFile"));
    menuBar->addAction(menuFile->menuAction());

    actionRenderScene1 = new QAction(this);
    actionRenderScene1->setObjectName(QStringLiteral("actionRenderScene1"));
    actionRenderScene1->setText(QApplication::translate("MCTracer", "Render Scene 1", Q_NULLPTR));
    actionRenderScene1->setShortcut(QApplication::translate("MCTracer", "Ctrl+1", Q_NULLPTR));
    menuFile->addAction(actionRenderScene1);
    actionRenderScene2 = new QAction(this);
    actionRenderScene2->setObjectName(QStringLiteral("actionRenderScene2"));
    actionRenderScene2->setText(QApplication::translate("MCTracer", "Render Scene 2", Q_NULLPTR));
    actionRenderScene2->setShortcut(QApplication::translate("MCTracer", "Ctrl+2", Q_NULLPTR));
    menuFile->addAction(actionRenderScene2);

    centralWidget = new QWidget(this);
    centralWidget->setObjectName(QStringLiteral("centralWidget"));
    this->setCentralWidget(centralWidget);

    screenLabel = new QLabel(centralWidget);
    screenLabel->setObjectName(QStringLiteral("screenLabel"));
    screenLabel->setGeometry(QRect(0, 0, 800, 600));
    screenLabel->setAlignment(Qt::AlignCenter);

    statusBar = new QStatusBar(this);
    statusBar->setObjectName(QStringLiteral("statusBar"));
    this->setStatusBar(statusBar);
}

void MCTracer::BindSignalSlot()
{
    connect(actionRenderScene1, &QAction::triggered, this, &MCTracer::RenderScene1);
    connect(actionRenderScene2, &QAction::triggered, this, &MCTracer::RenderScene2);
}

void MCTracer::RenderScene1()
{
    PW::FileReader::ObjModel model;
    model.readObj("Resources/scene01.obj");
    QImage image(200, 150, QImage::Format::Format_ARGB32);
    PWVector4f hostcolor[150][200]; // Width*Height
    PW::Tracer::RenderScene1(&model, &hostcolor[0][0]);
    for (int y = 0; y < 150; y++)
    {
        for (int x = 0; x < 200; x++)
        {
            PWVector4f &color = hostcolor[y][x];
            image.setPixelColor(QPoint(x, y), QColor(color.x * 255, color.y * 255, color.z * 255));
        }
    }
    screenLabel->setPixmap(QPixmap::fromImage(image));
}

void MCTracer::RenderScene2()
{
    PW::FileReader::ObjModel model;
    model.readObj("Resources/scene02.obj");
}