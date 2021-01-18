#include "mainwindow.h"
#include "ui_mainwindow.h"
#include"sstream"
#include<iostream>
#include<algorithm>
#include<vector>
#include<opencv2/opencv.hpp>
#include <opencv2/core//core.hpp>
#include<opencv2/imgcodecs/imgcodecs.hpp>
#include<opencv2/core/core.hpp>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
using namespace cv;
using namespace std;

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
}

MainWindow::~MainWindow()
{
    delete ui;
}




void MainWindow::on_pushButton_clicked()
{
    QString fileName = QFileDialog::getOpenFileName(this,
                                                    tr("Open Image"), ".", tr("Image Files (*.png *.jpg *.bmp)"));

    image_inp= cv::imread(fileName.toStdString());
    cv::cvtColor(image_inp,image_inp,CV_BGR2RGB);  // change color channel ordering
    cv::cvtColor(image_inp,image_inp_gray,CV_RGB2GRAY); // to change input image to grayscale
    cv::cvtColor(image_inp, hsvimage, CV_BGR2HSV);


    if (image_inp.data) {

        ui->process->setEnabled(true);
    }

    showInput(image_inp_gray);

}

void MainWindow::showInput(cv::Mat image_inp_gray){

    imageView = QImage((const unsigned char*)(image_inp_gray.data),  // Qt image structure
                       image_inp_gray.cols,image_inp_gray.rows,image_inp_gray.step,QImage::Format_Indexed8);

    // to know the label width and height to scale the image
    int width = ui->input->width();
    int height = ui->input->height();


    // convert QImage to QPixmap and show on QLabel
    ui->input->setPixmap(QPixmap::fromImage(imageView).scaled(width,height,Qt::KeepAspectRatio));
}

void MainWindow::on_process_clicked()
{

    cv::Mat blurr;
    cv::Mat canny;
    cv::Mat binary;
    cv::Mat output;
    cv::Mat output1;
    cv::Mat eroded;
    cv::Mat final;
    cv::Mat final1;
    cv::Mat dst;

    int MAX_KERNEL_LENGTH = 9;
    int DELAY_BLUR = 100;
    RNG rng(12345);
    vector<vector<Point> > contours;
    vector<vector<Point> > filtered_contours;
    std::vector<cv::Mat> planes;
    std::vector<cv::Mat> norm_img;
    vector<Vec4i> hierarchy;
    int morph_elem = 0;
    int morph_size = 0;
    int morph_operator = 0;
    int const max_operator = 4;
    int const max_elem = 2;
    int const max_kernel_size = 31;
    int g_kernel_shape = MORPH_ELLIPSE;
    int g_kernel_size = 3;
    int sum=0;
    int areaTH=12323145;


    //    cv::Canny(image_inp,contours,10,60);
    cv::threshold(image_inp, binary, 50, 255, cv::THRESH_BINARY);
    cv::split(binary,planes);
    cv::equalizeHist(planes[0], planes[1]);

    for(int i=0;i<planes.size();i++)
    {
        cv::dilate(planes[i],dst,cv::Mat(),Point(-1,-1),10);
        cv::erode(dst,dst,cv::Mat(),Point(-1,-1),10);
        cv::medianBlur(dst, blurr, 101 );
        cv::absdiff(planes[i], blurr, final);
        final1 = 255 - final;
        cv::normalize(final1, output, 0, 255, cv::NORM_MINMAX);
        norm_img.push_back(output);


    }
    cv::merge(norm_img,output1);
    cv::erode(output1,output1,cv::Mat(),Point(-1,-1),10);
    cv::imshow("image1",output1);
    cv::Canny(output1, canny, 20,20*2 );
    cv::imshow("image",canny);
    findContours( dst, contours, RETR_TREE, CHAIN_APPROX_SIMPLE );

    vector<Moments> mu(contours.size() );
    for( size_t i = 0; i < contours.size(); i++ )
    {
        mu[i] = moments( contours[i] );
    }
    vector<Point2f> mc( contours.size() );
    for( size_t i = 0; i < contours.size(); i++ )
    {
        //add 1e-5 to avoid division by zero
        mc[i] = Point2f( static_cast<float>(mu[i].m10 / (mu[i].m00 + 1e-5)),
                         static_cast<float>(mu[i].m01 / (mu[i].m00 + 1e-5)) );
        cout << "mc[" << i << "]=" << mc[i] << endl;
    }
    Mat drawing = Mat::zeros( canny.size(), CV_8UC3 );
    for( size_t i = 0; i< contours.size(); i++ )
    {
        Scalar color = Scalar( rng.uniform(0, 256), rng.uniform(0,256), rng.uniform(0,256) );
        drawContours( drawing,  contours, (int)i, color, 5, LINE_8, hierarchy, 0  );
        //        circle( drawing, mc[i], 4, color, -1 );

    }
    cout << "\t Info: Area and Contour Length \n";
    for( size_t i = 0; i < contours.size(); i++ )
    {
        cout << " * Contour[" << i << "] - Area (M_00) = " << std::fixed << std::setprecision(2) << mu[i].m00
             << " - Area OpenCV: " << contourArea(contours[i]) << " - Length: " << arcLength( contours[i], true ) << endl;
        if(contourArea(contours[i])>=areaTH)
        {
            sum=sum+1;
        }

    }


    ui->count->clear();
    ui->count->append(QString::number(sum));


    //    Mat drawing = Mat::zeros( dst.size(), CV_8UC3 );
    //    for( size_t i = 0; i< contours.size(); i++ )
    //    {
    //        Moments moms = moments(contours[i]);
    //        double area = moms.m00;
    //        double perimeter = arcLength(contours[i], true);
    //        double ratio = 4 * CV_PI * area / (perimeter * perimeter);
    //        if (ratio >= 0.00000001)
    //            filtered_contours.push_back(contours[i]);
    //        Scalar color = Scalar( rng.uniform(0, 256), rng.uniform(0,256), rng.uniform(0,256) );
    //        drawContours( drawing, contours, (int)i, color, 2, LINE_8, hierarchy, 0 );

    //        drawContours( drawing, filtered_contours,  (int)i, color, 2, LINE_8, hierarchy, 0  );

    //    }





    // cv::threshold(output1, binary, 100, 255, cv::THRESH_BINARY);


    //            cv::Canny(output1, canny, 120,115*2 );



    //         int operation = morph_operator+ 2;
    //         Mat element = getStructuringElement( morph_elem, Size( 2*morph_size + 1, 2*morph_size+1 ), Point( morph_size, morph_size ) );
    //         morphologyEx( canny, dst, operation, element );

    //            cv::findContours(canny,contours,cv::RETR_LIST,cv::CHAIN_APPROX_NONE); // all pixels
    //            vector<vector<Point> > contours_poly( contours.size() );
    //            vector<Rect> boundRect( contours.size() );
    //            vector<Point2f>centers( contours.size() );
    //            vector<float>radius( contours.size() );
    //            for( size_t i = 0; i < contours.size(); i++ )
    //            {
    //                approxPolyDP( contours[i], contours_poly[i], 3, true );
    //                boundRect[i] = boundingRect( contours_poly[i] );
    //                minEnclosingCircle( contours_poly[i], centers[i], radius[i] );
    //            }
    //            Mat drawing = Mat::zeros( canny.size(), CV_8UC3 );
    //            for( size_t i = 0; i< contours.size(); i++ )
    //            {
    //                Scalar color = Scalar( rng.uniform(0, 256), rng.uniform(0,256), rng.uniform(0,256) );


    //                circle( drawing, centers[i], (int)radius[i], color, 2 );
    //            }




    showOutput(drawing);


    //*/     perform gaussian blur
    //        cv::GaussianBlur(image_inp_gray, imageBlur, cv::Size(5, 5), 1.5);
    //        vector<Vec3f> circles;
    //            cv::HoughCircles(image_inp_gray, circles, CV_HOUGH_GRADIENT, 1,
    //                         image_inp_gray.rows/16,  // change this value to detect circles with different distances to each other
    //                         120,50 , 200, 350// change the last two parameters
    //                    // (min_radius & max_radius) to detect larger circles
    //            );
    //            for( size_t i = 0; i < circles.size(); i++ )
    //            {
    //                Vec3i c = circles[i];
    //                Point center = Point(c[0], c[1]);
    //                // circle center
    //                cv::circle( image_inp_gray, center, 1, Scalar(0,100,100), 3);
    //                // circle outline
    //                int radius = c[2];
    //                cv::circle( image_inp_gray, center, radius, Scalar(255,0,255), 3);
    //            }



    //         showOutput(image_inp_gray);

}

void MainWindow::showOutput(cv::Mat image){

    ui->output->clear();

    if(image.channels() == 1){
        imageView = QImage((const unsigned char*)(image.data),  // Qt image structure
                           image.cols,image.rows,image.step,QImage::Format_Indexed8);
    }
    else{
        imageView = QImage((const unsigned char*)(image.data),  // Qt image structure
                           image.cols,image.rows,image.step,QImage::Format_RGB888);
    }


    // to know the label width and height to scale the image
    int width = ui->output->width();
    int height = ui->output->height();





    // convert QImage to QPixmap and show on QLabel
    ui->output->setPixmap(QPixmap::fromImage(imageView).scaled(width,height,Qt::KeepAspectRatio));
}
