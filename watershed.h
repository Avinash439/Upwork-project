#ifndef WATERSHED_H
#define WATERSHED_H

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

class WatershedSegmenter{
private:

public:
    cv::Mat markers;

    void setMarkers(cv::Mat& markerImage)
    {

        markerImage.convertTo(markers, CV_32S);
    }

    cv::Mat process(cv::Mat &image)
    {
        cv::watershed(image, markers);
        markers.convertTo(markers,CV_8U);
        return markers;
    }

};

#endif // WATERSHED_H
