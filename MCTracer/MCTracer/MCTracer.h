#pragma once

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
    QAction *actionLoadScene;

    QWidget *centralWidget;
    QStatusBar *statusBar;

#pragma endregion

    public slots:
    void LoadScene();
};
