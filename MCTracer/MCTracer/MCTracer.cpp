#include "MCTracer.h"

#include <QtWidgets/QApplication>

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

    actionLoadScene = new QAction(this);
    actionLoadScene->setObjectName(QStringLiteral("actionLoadScene"));
    actionLoadScene->setText(QApplication::translate("MCTracer", "Load Scene", Q_NULLPTR));
    actionLoadScene->setShortcut(QApplication::translate("MCTracer", "Ctrl+O", Q_NULLPTR));
    menuFile->addAction(actionLoadScene);

    centralWidget = new QWidget(this);
    centralWidget->setObjectName(QStringLiteral("centralWidget"));
    this->setCentralWidget(centralWidget);

    statusBar = new QStatusBar(this);
    statusBar->setObjectName(QStringLiteral("statusBar"));
    this->setStatusBar(statusBar);
}

void MCTracer::BindSignalSlot()
{
    connect(actionLoadScene, &QAction::triggered, this, &MCTracer::LoadScene);
}

void MCTracer::LoadScene()
{
    PW::FileReader::ObjModel model;
    model.readObj("Resources/scene01.obj");
    statusBar->showMessage("Load Scene");
}