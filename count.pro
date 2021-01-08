#-------------------------------------------------
#
# Project created by QtCreator 2021-01-05T18:48:50
#
#-------------------------------------------------

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = count
TEMPLATE = app



SOURCES += main.cpp\
        mainwindow.cpp


HEADERS  += mainwindow.h \
    watershed.h

FORMS    += mainwindow.ui


INCLUDEPATH += /home/avinash/upwork/opencv-3.2.0/build/include
LIBS += -L"/home/avinash/upwork/opencv-3.2.0/build/lib"
LIBS += -lopencv_calib3d
LIBS += -lopencv_core
LIBS += -lopencv_features2d
LIBS += -lopencv_flann
LIBS += -lopencv_highgui
LIBS += -lopencv_imgproc
LIBS += -lopencv_ml
LIBS += -lopencv_objdetect
LIBS += -lopencv_photo
LIBS += -lopencv_stitching
LIBS += -lopencv_superres
