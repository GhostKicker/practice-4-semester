#include "myheader.h"

const int inf = static_cast<int>(1e9) + 7;
const int start_frame = 100;
const int finish_frame = 200;

Mat dilate_element = getStructuringElement(MORPH_RECT, Size(7, 7), Point(3, 3));

string filename = "C:\\Users\\User\\Desktop\\OS\\vid1.avi";

double dist2(pillar& a, pillar& b)
{
    return (pow(a.xpos_av - b.xpos_av, 2) + pow(a.ypos_av - b.ypos_av, 2));
}


int main() 
{
    VideoCapture vid(filename.c_str());

    if (!vid.isOpened())
    {
        cout << "Cannot read file";
        return 1;
    }
    cout << "opened successfully" << endl;

    //create array with all frames in video
    ptrdiff_t current_frame_index = 0;
    ptrdiff_t count = 0;
    vector<Mat> frames;
    frames.push_back(Mat());
    while (vid.read(frames[current_frame_index - count]) && current_frame_index <= finish_frame)
    {
        if (current_frame_index < start_frame)
        {
            frames.pop_back();
            ++count;
        }
        frames.push_back(Mat());
        cout << ++current_frame_index << " read" << endl;
    }
    frames.pop_back();

    int n = frames.size();


    vector<Mat> grayframes(n);
    vector<UMat> flow_vector(n);
    vector<vector<UMat> > flow_splitted(n);
    vector<Mat> outflow(n);
    vector<Mat> canny(n);
    vector<Mat> flow_integrals(n);
    vector<Mat> canny_integrals(n);
    vector<vector<Vec4i> > lines(n);

    //what pillars there are on i-th frame
    vector<vector<pillar> > pillars(n);
    vector<vector<Rect> > bounding_rects(n);
    vector<Mat> drawings(n);


    vector<Mat> dilated_cannys(n);
    vector<Mat> good_lines_from_cannys(n);

    //Mat rects(rect_rows, rect_cols, 0);

    Size small_size = fitSize(frames[0].size(), Size(frame_size.first, frame_size.second));

    //resize, transfer all frames to gray
    for (int i = 0; i < n; ++i)
    {
        resize(frames[i], frames[i], small_size);
        cvtColor(frames[i], grayframes[i], COLOR_BGR2GRAY);
    }
    cout << "transfered to gray" << endl;


    //visible flow
    cv::Ptr<cv::DenseOpticalFlow> alg = cv::FarnebackOpticalFlow::create();
    cout << "started to calc flow" << endl;
    for (int i = 1; i < n; ++i)
    {
        alg->calc(grayframes[i - 1], grayframes[i], flow_vector[i]);
        outflow[i] = getVisibleFlow(flow_vector[i]);
        split(flow_vector[i], flow_splitted[i]);
        cout << i << " / " << n << " calced flow" << endl;
    }


    //Cannys edge detector
    for (int i = 1; i < n; ++i)
    {
        CannyFull(frames[i], grayframes[i], canny[i]);
        cout << i << "/" << n << " canny" << endl;
    }

    //dilate cannys
    for (int i = 1; i < n; ++i)
    {
        dilate(canny[i], dilated_cannys[i], dilate_element);
        cout << i << "/" << n << " dilate" << endl;
    }

    //get lines
    for (int i = 1; i < n; ++i)
    {
        BlackImage.copyTo(good_lines_from_cannys[i]);
        get_lines_from_canny(dilated_cannys[i], good_lines_from_cannys[i], 40);
        cout << i << "/" << n << " good lines" << endl;
    }

    //calculate pillars pos
    for (int i = 1; i < n; i++)
    {
        Mat tmp(WhiteImage.size(), 4, 10000);

        //for every row
        for (int currow = 0; currow < frame_size.second; ++currow)
        {   //for every col
            for (int curcol = 0; curcol < frame_size.first; ++curcol)
            {   //if is white pixel that doesn't belong to any of pillars
                if (good_lines_from_cannys[i].at<uchar>(currow, curcol) != 1)
                if (good_lines_from_cannys[i].at<uchar>(currow, curcol) != 1 && tmp.at<int>(currow, curcol) == 10000)
                {
                    //bfs to fill up all pixels of that column
                    queue<Point2i> q;
                    q.push(Point2i(curcol, currow));
                    int curindex = pillars[i].size();
                    pillars[i].push_back(pillar());
                    while (!q.empty())
                    {
                        auto curpt = q.front();
                        q.pop();
                        if (tmp.at<int>(curpt) != 10000) continue;
                        pillars[i][curindex].points.push_back(curpt);
                        tmp.at<int>(curpt) = curindex;

                        Point2i nextpt = (curpt + Point2i(-1, 0)); //left
                        if (curpt.x > 0 && tmp.at<int>(nextpt) == 10000 && good_lines_from_cannys[i].at<uchar>(nextpt) != 1) q.push(nextpt);
                        nextpt = (curpt + Point2i(0, -1)); //top
                        if (curpt.y > 0 && tmp.at<int>(nextpt) == 10000 && good_lines_from_cannys[i].at<uchar>(nextpt) != 1) q.push(nextpt);
                        nextpt = (curpt + Point2i(1, 0)); //right
                        if (curpt.x < frame_size.first - 1 && tmp.at<int>(nextpt) == 10000 && good_lines_from_cannys[i].at<uchar>(nextpt) != 1) q.push(nextpt);
                        nextpt = (curpt + Point2i(0, 1)); //bot
                        if (curpt.y < frame_size.second - 1 && tmp.at<int>(nextpt) == 10000 && good_lines_from_cannys[i].at<uchar>(nextpt) != 1) q.push(nextpt);
                    }
                }
            }
        }
        cout << i << "/" << n << " analyzed pillars" << endl;
    }

    //get bounding rects
    for (int i = 1; i < n; ++i)
    {
        int cnt = 0;
        for (int j = 0; j < pillars[i].size() - cnt; j++)
        {
            bounding_rects[i].push_back(boundingRect(Mat(pillars[i][j].points)));
            if (bounding_rects[i][j].area() < 150 || bounding_rects[i][j].height < 70)
            {
                bounding_rects[i].pop_back();
                pillars[i].erase(pillars[i].begin() + j);
                ++cnt;
                --j;
            }
        }
    }

    //calculate pos, d
    for (int j = 1; j < n; j++)
    {
        for (int i = 0; i < bounding_rects[j].size(); i++)
        {
            Mat& x_flow = flow_splitted[j][0].getMat(ACCESS_READ);
            Mat& y_flow = flow_splitted[j][1].getMat(ACCESS_READ);
            pillars[j][i].calculate_pos();
            pillars[j][i].calculate_d(x_flow, y_flow);
        }
        cout << j << " / " << n << " calc pos'es & d's" << endl;
    }

    //match pillars of i-th and (i+1)-th frame
    for (int i = 1; i < n - 1; i++)
    {
        for (int j = 0; j < pillars[i].size(); ++j)
        {
            pillar& curpl = pillars[i][j];
            double curdist2 = inf;
            int curnext = -1;

            for (int jj = 0; jj < pillars[i + 1].size(); ++jj)
            {
                pillar& nextpl = pillars[i + 1][jj];

                if (dist2(curpl, nextpl) < 150 + 4 * curpl.line_length2() && dist2(curpl, nextpl) < curdist2)
                {
                    curnext = jj;
                }
            }
            
            if (curnext != -1)
            {
                pillars[i + 1][curnext].color = curpl.color;
            }
        }
        cout << i << " / " << n << " calc nexts" << endl;
    }


    //draw
    for (int j = 1; j < n; ++j)
    {
        drawings[j] = Mat::zeros(frames[j].size(), CV_8UC3);
        for (int i = 0; i< bounding_rects[j].size(); i++)
        {
            rectangle(drawings[j], bounding_rects[j][i], pillars[j][i].color, 2);
            rectangle(drawings[j], Rect(pillars[j][i].xpos_av, pillars[j][i].ypos_av, 1, 1), pillars[j][i].color, 3);

            if (pillars[j][i].xpos_av + pillars[j][i].dx_av < frame_size.first && pillars[j][i].xpos_av + pillars[j][i].dx_av < frame_size.first )
                line(drawings[j], Point(pillars[j][i].xpos_av, pillars[j][i].ypos_av), Point(pillars[j][i].xpos_av + pillars[j][i].dx_av, pillars[j][i].ypos_av + pillars[j][i].dy_av), pillars[j][i].color, 2);
        }
    }


    //cycle for showing frames
    //press 'd' to move to next frame
    //press 'a' to move to previous frame
    //press 'ESC' to exit
    current_frame_index = 1;
    ptrdiff_t left_border = 1;
    ptrdiff_t right_border = frames.size() - 1;
    while (true)
    {
        imshow("frames", frames[current_frame_index]);
        imshow("grayframes", grayframes[current_frame_index]);
        imshow("flow", outflow[current_frame_index]);
        imshow("canny_edges", canny[current_frame_index]);
        imshow("dilated cannys", dilated_cannys[current_frame_index]);
        imshow("good lines", good_lines_from_cannys[current_frame_index]);
        imshow("rects", drawings[current_frame_index]);

        int current_key = waitKey(10);
        if (current_key == 100 && current_frame_index < right_border)
            ++current_frame_index;
        else if (current_key == 97 && current_frame_index > left_border)
            --current_frame_index;
        else if (current_key == 27)
            break;
        cout << current_frame_index << endl;
    }

    return 0;
}
