/********************************************************************************
** Form generated from reading UI file 'MCTracer.ui'
**
** Created by: Qt User Interface Compiler version 5.10.1
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_MCTRACER_H
#define UI_MCTRACER_H

#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QButtonGroup>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QMenuBar>
#include <QtWidgets/QStatusBar>
#include <QtWidgets/QToolBar>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_MCTracerClass
{
public:
    QMenuBar *menuBar;
    QToolBar *mainToolBar;
    QWidget *centralWidget;
    QStatusBar *statusBar;

    void setupUi(QMainWindow *MCTracerClass)
    {
        if (MCTracerClass->objectName().isEmpty())
            MCTracerClass->setObjectName(QStringLiteral("MCTracerClass"));
        MCTracerClass->resize(600, 400);
        menuBar = new QMenuBar(MCTracerClass);
        menuBar->setObjectName(QStringLiteral("menuBar"));
        MCTracerClass->setMenuBar(menuBar);
        mainToolBar = new QToolBar(MCTracerClass);
        mainToolBar->setObjectName(QStringLiteral("mainToolBar"));
        MCTracerClass->addToolBar(mainToolBar);
        centralWidget = new QWidget(MCTracerClass);
        centralWidget->setObjectName(QStringLiteral("centralWidget"));
        MCTracerClass->setCentralWidget(centralWidget);
        statusBar = new QStatusBar(MCTracerClass);
        statusBar->setObjectName(QStringLiteral("statusBar"));
        MCTracerClass->setStatusBar(statusBar);

        retranslateUi(MCTracerClass);

        QMetaObject::connectSlotsByName(MCTracerClass);
    } // setupUi

    void retranslateUi(QMainWindow *MCTracerClass)
    {
        MCTracerClass->setWindowTitle(QApplication::translate("MCTracerClass", "MCTracer", nullptr));
    } // retranslateUi

};

namespace Ui {
    class MCTracerClass: public Ui_MCTracerClass {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_MCTRACER_H
