#pragma once

#include <QtWidgets/QLabel>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QMenuBar>
#include <QtWidgets/QStatusBar>
#include <QtWidgets/QWidget>

class MCTracer : public QMainWindow
{
    Q_OBJECT

public:
    MCTracer(QWidget *parent = Q_NULLPTR);

private:
    void InitUI();
    void BindSignalSlot();

#pragma region UI

    QMenuBar *menuBar;
    QMenu *menuFile;
    QAction *actionRenderScene1;
    QAction *actionRenderScene2;

    QWidget *centralWidget;
    QLabel *screenLabel;
    QStatusBar *statusBar;

#pragma endregion

    public slots:
    void RenderScene1();
    void RenderScene2();
};
