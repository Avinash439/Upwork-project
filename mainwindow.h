#ifndef MAINWINDOW_H
#define MAINWINDOW_H
#include <iostream>
#include <QMainWindow>
#include <QFileDialog>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "watershed.h"

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();

    cv::Mat image_inp; // input image varaiable
    cv::Mat image_inp_gray;
    cv::Mat image_out;
    cv::Mat hsvimage;
    cv::Mat  imageBlur;
    QImage imageView;
    cv::Mat markers;
    std::vector<cv::Vec3f> circles;



    void showInput(cv::Mat image_inp_gray); // to show input image
    void showOutput(cv::Mat); // to show output image
    int display_dst( int delay );



private slots:
    void on_pushButton_clicked();



    void on_process_clicked();

private:
    Ui::MainWindow *ui;
    WatershedSegmenter segmenter;
};

#endif // MAINWINDOW_H
