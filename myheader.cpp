#include "myheader.h";

Size fitSize(const Size & sz, const Size & bounds)
{
    CV_Assert(sz.area() > 0);
    if (sz.width > bounds.width || sz.height > bounds.height)
    {
        double scale = std::min((double)bounds.width / sz.width, (double)bounds.height / sz.height);
        return Size(cvRound(sz.width * scale), cvRound(sz.height * scale));
    }
    return sz;
}

Mat getVisibleFlow(InputArray flow)
{
    vector<UMat> flow_vec;
    split(flow, flow_vec);
    UMat magnitude, angle;
    cartToPolar(flow_vec[0], flow_vec[1], magnitude, angle, true);
    magnitude.convertTo(magnitude, CV_32F, 0.2);
    vector<UMat> hsv_vec;
    hsv_vec.push_back(angle);
    hsv_vec.push_back(UMat::ones(angle.size(), angle.type()));
    hsv_vec.push_back(magnitude);
    UMat hsv;
    merge(hsv_vec, hsv);
    Mat img;
    cvtColor(hsv, img, COLOR_HSV2BGR);
    return img;
}


void CannyFull(Mat& src, Mat& src_gray, Mat& dst)
{
    dst.create(src.size(), src.type());
    Mat detected_edges;
    int edgeThresh = 1;
    int lowThreshold = 100;
    int ratio = 3;
    int kernel_size = 3;

    CannyThreshold(0, 0, src, src_gray, detected_edges, lowThreshold, ratio, kernel_size, dst);
}

void CannyThreshold(int, void*, Mat& src, Mat& src_gray, Mat& detected_edges, int lowThreshold, int ratio, int kernel_size, Mat& dst)
{
    /// Reduce noise with a kernel 3x3
    blur(src_gray, detected_edges, Size(3, 3));

    /// Canny detector
    Canny(detected_edges, detected_edges, lowThreshold, lowThreshold*ratio, kernel_size);

    /// Using Canny's output as a mask, we display our result
    dst = Scalar::all(0);

    WhiteImage.copyTo(dst, detected_edges);
}


void get_lines_from_canny(Mat& canny, Mat& lines, int limit)
{
    for (int col = 0; col < frame_size.first; ++col)
    {
        for (int row = 0; row < frame_size.second; ++row)
        {
            if (!canny.at<uchar>(row, col)) continue;
            int curcnt = 1;
            for (int i = 1; i < limit; ++i)
            {
                if (curcnt == limit) break;
                if (row - i < 0) break;
                if (!canny.at<uchar>(row - i, col)) break;
                ++curcnt;
            }
            for (int i = 1; i < limit; ++i)
            {
                if (curcnt == limit) break;
                if (row + i >= frame_size.second) break;
                if (!canny.at<uchar>(row + i, col)) break;
                ++curcnt;
            }
            if (curcnt >= limit)
                lines.at<uchar>(row, col) = canny.at<uchar>(row, col);
        }
    }
}



