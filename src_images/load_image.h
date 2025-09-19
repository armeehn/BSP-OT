#pragma once

#include "../src/BSPOT.h"

#include <fstream>
#include <iostream>
#include <filesystem>
#include <spdlog/spdlog.h>

#include "RGB_to_XYZ.hpp"
#include "XYZ_to_Lab.hpp"

namespace BSPOT {

inline fs::path convertToPPM(fs::path path) {
    // remove extension from path
    auto out_path = path;
    out_path.replace_extension(".ppm");
    std::ifstream in(path);
    std::ofstream out(out_path);
    std::string command = "convert " + path.string() + " -compress None " + out_path.string();
    spdlog::info("command: {}",command);
    system(command.c_str());
    return out_path;
}

inline fs::path removeComments(fs::path path) {
    //read file and remove all lines that start with #
    std::ifstream in(path);
    std::ofstream out("/tmp/buffer_comments.ppm");
    std::string line;
    while (std::getline(in,line)) {
        if (line[0] != '#')
            out << line << "\n";
    }
    return fs::path("/tmp/buffer_comments.ppm");
}

using Pixels = Eigen::Matrix<double,3,-1>;
using Pixel = Eigen::Vector3d;

inline Pixels LoadImageRGB(fs::path path,int& W,int& H) {
    auto ppm_path = convertToPPM(path);
    auto ppm_ok = removeComments(ppm_path);
    std::ifstream file(ppm_ok);
    std::string tmp;
    file >> tmp;
    file >> W >> H;
    file >> tmp;
    Pixels rslt = Pixels::Zero(3,W*H);
    for (auto i : range(W*H)){
        Pixel x;
        file >> x(0) >> x(1) >> x(2);
        rslt.col(i) = x;
    }
    rslt /= 255;
    return rslt;
}

inline Vector<-1> Luminance(const Pixels& colors) {
    Vector<-1> L(colors.cols());
    for (auto i : range(colors.cols())) {
        Pixel c = colors.col(i);
        L[i] = 0.2126*c(0) + 0.7152*c(1) + 0.0722*c(2);
    }
    return L;
}

inline fs::path ResizeTo(fs::path path,int W,int H,bool keep_ratio = false) {
    // keep same name and extension but place it in /tmp
    auto filename = path.filename();
    auto out = fs::path("/tmp/buffer.ppm");
    out.replace_extension(path.extension());

    std::string command;
    if (keep_ratio)
        command = "convert " + path.string() + " -resize " + std::to_string(W) + "x" + std::to_string(H) + " " + out.string();
    else
        command = "convert " + path.string() + " -resize " + std::to_string(W) + "x" + std::to_string(H) + "\\! " + out.string();
    spdlog::info("command: {}",command);
    system(command.c_str());
    return out;
}

inline void exportImageRGB(fs::path path, const Points<3>& X,int W,int H) {
    std::string buffer("/tmp/rslt.ppm");
    std::ofstream file(buffer);
    file << "P3\n";
    file << W << " " << H << "\n";
    file << "255\n";
    for (auto i : range(X.cols())) {
        auto pixel = X.col(i);
        int r = static_cast<int>(std::clamp<float>(pixel(0),0.,1.) * 255.0f);
        int g = static_cast<int>(std::clamp<float>(pixel(1),0.,1.) * 255.0f);
        int b = static_cast<int>(std::clamp<float>(pixel(2),0.,1.) * 255.0f);
        file << r << " " << g << " " << b << std::endl;
    }
    file.close();
    std::string command = "convert " + buffer + " " + path.string();
    spdlog::info("command: {}",command);
    system(command.c_str());
}

inline Pixels RGBtoLab(Pixels Img) {
    using namespace colorutil;
    for (auto i : range(Img.cols()))
        Img.col(i) = convert_XYZ_to_Lab(convert_RGB_to_XYZ(Img.col(i)));
    return Img;
}

}
