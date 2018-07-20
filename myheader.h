#ifndef MYHEADER_H
#define MYHEADER_H

#define _CRT_SECURE_NO_WARNINGS

#include <iostream>
#include <vector>
#include <queue>

#include "opencv2/core/ocl.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/video.hpp"

using namespace std;
using namespace cv;

const pair<int, int> rect_size(10, 100);
const pair<int, int> frame_size(800, 450);

const Mat WhiteImage(Size(frame_size.first, frame_size.second), 0, 255);
const Mat BlackImage(Size(frame_size.first, frame_size.second), 0, 1);

const int rect_rows = frame_size.second / rect_size.second + 1;
const int rect_cols = frame_size.first / rect_size.first + 1;

const double pixels_per_row = (frame_size.second - 1) / double(rect_rows);
const double pixels_per_col = (frame_size.first - 1) / double(rect_cols);

static RNG rng(12345);

Size fitSize(const Size & sz, const Size & bounds);
Mat getVisibleFlow(InputArray flow);
void CannyThreshold(int, void*, Mat& src, Mat& src_gray, Mat& detected_edges, int lowThreshold, int ratio, int kernel_size, Mat& dst);
void CannyFull(Mat& src, Mat& src_gray, Mat& dst);
void get_lines_from_canny(Mat& canny, Mat& lines, int limit);


template <typename T>
T getIntegralInRect(int x1, int y1, int x2, int y2, Mat& src)
{
    return src.at<T>(y2, x2) - src.at<T>(y2, x1) - src.at<T>(y1, x2) + src.at<T>(y1, x1);
}

struct pillar
{
    vector<Point2i> points;
    double xpos_av = 0;
    double ypos_av = 0;
    double dx_av = 0;
    double dy_av = 0;
    Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
    pillar() {};
    ~pillar() {};

    void calculate_pos()
    {
        xpos_av = 0;
        ypos_av = 0;
        for (auto curpt : points)
        {
            xpos_av += curpt.x;
            ypos_av += curpt.y;
        }
        xpos_av /= points.size();
        ypos_av /= points.size();
    }

    void calculate_d(Mat& x_flow, Mat& y_flow)
    {
        dx_av = 0;
        dy_av = 0;
        for (auto pt : points)
        {
            dx_av += x_flow.at<float>(pt);
            dy_av += y_flow.at<float>(pt);
        }
        dx_av /= points.size();
        dy_av /= points.size();
    }


    double line_length2()
    {
        return dx_av * dx_av + dy_av * dy_av;
    }
};



#endif