// clang-format off
//
// Created by goksu on 4/6/19.
//

#include <algorithm>
#include <vector>
#include "rasterizer.hpp"
#include <opencv2/opencv.hpp>
#include <math.h>


rst::pos_buf_id rst::rasterizer::load_positions(const std::vector<Eigen::Vector3f> &positions)
{
    auto id = get_next_id();
    pos_buf.emplace(id, positions);

    return {id};
}

rst::ind_buf_id rst::rasterizer::load_indices(const std::vector<Eigen::Vector3i> &indices)
{
    auto id = get_next_id();
    ind_buf.emplace(id, indices);

    return {id};
}

rst::col_buf_id rst::rasterizer::load_colors(const std::vector<Eigen::Vector3f> &cols)
{
    auto id = get_next_id();
    col_buf.emplace(id, cols);

    return {id};
}

auto to_vec4(const Eigen::Vector3f& v3, float w = 1.0f)
{
    return Vector4f(v3.x(), v3.y(), v3.z(), w);
}


static bool insideTriangle(float x, float y, const Vector3f* _v)
{   
    // Implement this function to check if the point (x, y) is inside the triangle represented by _v[0], _v[1], _v[2]
    Vector2f screen_point = Vector2f(x, y);

    Vector2f v[3];
    v[0] = _v[1].head(2) - _v[0].head(2);
    v[1] = _v[2].head(2) - _v[1].head(2);
    v[2] = _v[0].head(2) - _v[2].head(2);

    Vector2f dir[3];
    dir[0] = screen_point - _v[0].head(2);
    dir[1] = screen_point - _v[1].head(2);
    dir[2] = screen_point - _v[2].head(2);

    int count = 0;
    for(int i = 0; i < 3; ++i)
    {
        float result = v[i][0] * dir[i][1] - v[i][1] * dir[i][0];
        if(result > 0)
        {
            count++;
        }
        else if(result < 0)
        {
            count--;
        }
    }

    return count == 3 || count == -3;
}

static std::tuple<float, float, float> computeBarycentric2D(float x, float y, const Vector3f* v)
{
    float c1 = (x*(v[1].y() - v[2].y()) + (v[2].x() - v[1].x())*y + v[1].x()*v[2].y() - v[2].x()*v[1].y()) / (v[0].x()*(v[1].y() - v[2].y()) + (v[2].x() - v[1].x())*v[0].y() + v[1].x()*v[2].y() - v[2].x()*v[1].y());
    float c2 = (x*(v[2].y() - v[0].y()) + (v[0].x() - v[2].x())*y + v[2].x()*v[0].y() - v[0].x()*v[2].y()) / (v[1].x()*(v[2].y() - v[0].y()) + (v[0].x() - v[2].x())*v[1].y() + v[2].x()*v[0].y() - v[0].x()*v[2].y());
    float c3 = (x*(v[0].y() - v[1].y()) + (v[1].x() - v[0].x())*y + v[0].x()*v[1].y() - v[1].x()*v[0].y()) / (v[2].x()*(v[0].y() - v[1].y()) + (v[1].x() - v[0].x())*v[2].y() + v[0].x()*v[1].y() - v[1].x()*v[0].y());
    return {c1,c2,c3};
}

void rst::rasterizer::draw(pos_buf_id pos_buffer, ind_buf_id ind_buffer, col_buf_id col_buffer, Primitive type)
{
    auto& buf = pos_buf[pos_buffer.pos_id];
    auto& ind = ind_buf[ind_buffer.ind_id];
    auto& col = col_buf[col_buffer.col_id];

    float f1 = (50 - 0.1) / 2.0;
    float f2 = (50 + 0.1) / 2.0;

    Eigen::Matrix4f mvp = projection * view * model;
    for (auto& i : ind)
    {
        Triangle t;
        Eigen::Vector4f v[] = {
                mvp * to_vec4(buf[i[0]], 1.0f),
                mvp * to_vec4(buf[i[1]], 1.0f),
                mvp * to_vec4(buf[i[2]], 1.0f)
        };
        //Homogeneous division
        for (auto& vec : v) {
            vec /= vec.w();
        }
        //Viewport transformation
        for (auto & vert : v)
        {
            vert.x() = 0.5*width*(vert.x()+1.0);
            vert.y() = 0.5*height*(vert.y()+1.0);
            vert.z() = vert.z() * f1 + f2;
        }

        for (int i = 0; i < 3; ++i)
        {
            t.setVertex(i, v[i].head<3>());
        }

        auto col_x = col[i[0]];
        auto col_y = col[i[1]];
        auto col_z = col[i[2]];

        t.setColor(0, col_x[0], col_x[1], col_x[2]);
        t.setColor(1, col_y[0], col_y[1], col_y[2]);
        t.setColor(2, col_z[0], col_z[1], col_z[2]);

        rasterize_triangle(t);
    }
}

//Screen space rasterization
void rst::rasterizer::rasterize_triangle(const Triangle& t) {
    auto v = t.toVector4();
    
    // Find out the bounding box of current triangle.
    // iterate through the pixel and find if the current pixel is inside the triangle
    Vector2i minAabb; // left bottom
    Vector2i maxAabb; // right up

    minAabb[0] = std::min(v[0][0],std::min(v[1][0], v[2][0]));
    maxAabb[0] = std::max(v[0][0],std::max(v[1][0], v[2][0]));
    minAabb[1] = std::min(v[0][1],std::min(v[1][1], v[2][1]));
    maxAabb[1] = std::max(v[0][1],std::max(v[1][1], v[2][1]));

    if(minAabb[0] > 0)
        minAabb[0] -= 1;
    if(minAabb[1] > 0)
        minAabb[1] -= 1;
    if(maxAabb[0] < width - 1)
        maxAabb[0] += 1;
    if(maxAabb[1] < height - 1)
        maxAabb[1] += 1;

    bool MSAA = false;

    std::vector<Eigen::Vector2f> pos;

    if(MSAA)
    {
        pos.push_back({0.25,0.25});
        pos.push_back({0.75,0.25});
        pos.push_back({0.25,0.75});
        pos.push_back({0.75,0.75});
    }
    else
    {
        pos.push_back({0.5, 0.5});
    }

    for(int x = minAabb[0]; x < maxAabb[0]; ++x)
    {
        for(int y = minAabb[1]; y < maxAabb[1]; ++y)
        {
            // 记录最小深度
            float minDepth = FLT_MAX;
            // 四个小点中落入三角形中的点的个数
            int count = 0;
            // 对四个小点坐标进行判断 
            for (int i = 0; i < pos.size(); i++)
            {
                // 小点是否在三角形内
                if (insideTriangle((float)x + pos[i][0], (float)y + pos[i][1], t.v))
                {
                    // 如果在，对深度z进行插值
                    auto[alpha, beta, gamma] = computeBarycentric2D((float)x + pos[i][0], (float)y + pos[i][1], t.v);
                    float w_reciprocal = 1.0 / (alpha / v[0].w() + beta / v[1].w() + gamma / v[2].w());
                    float z_interpolated = alpha * v[0].z() / v[0].w() + beta * v[1].z() / v[1].w() + gamma * v[2].z() / v[2].w();
                    z_interpolated *= w_reciprocal;
                    minDepth = std::min(minDepth, z_interpolated);
                    count++;
                }
            }
            if (count != 0)
            {
                if (depth_buf[get_index(x, y)] > minDepth)
                {
                    Vector3f color = t.getColor() * count / (float)pos.size();
                    Vector3f point(3);
                    point << (float)x, (float)y, minDepth;
                    // 替换深度
                    depth_buf[get_index(x, y)] = minDepth;
                    // 修改颜色
                    set_pixel(point, color);
                }
            }
        }
    }
}

void rst::rasterizer::set_model(const Eigen::Matrix4f& m)
{
    model = m;
}

void rst::rasterizer::set_view(const Eigen::Matrix4f& v)
{
    view = v;
}

void rst::rasterizer::set_projection(const Eigen::Matrix4f& p)
{
    projection = p;
}

void rst::rasterizer::clear(rst::Buffers buff)
{
    if ((buff & rst::Buffers::Color) == rst::Buffers::Color)
    {
        std::fill(frame_buf.begin(), frame_buf.end(), Eigen::Vector3f{0, 0, 0});
    }
    if ((buff & rst::Buffers::Depth) == rst::Buffers::Depth)
    {
        std::fill(depth_buf.begin(), depth_buf.end(), std::numeric_limits<float>::infinity());
    }
}

rst::rasterizer::rasterizer(int w, int h) : width(w), height(h)
{
    frame_buf.resize(w * h);
    depth_buf.resize(w * h);
}

int rst::rasterizer::get_index(int x, int y)
{
    return (height-1-y)*width + x;
}

void rst::rasterizer::set_pixel(const Eigen::Vector3f& point, const Eigen::Vector3f& color)
{
    //old index: auto ind = point.y() + point.x() * width;
    auto ind = (height-1-point.y())*width + point.x();
    frame_buf[ind] = color;

}

// clang-format on