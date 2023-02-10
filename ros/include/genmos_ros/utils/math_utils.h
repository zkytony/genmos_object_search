#ifndef MY_MATH_UTILS_H
#define MY_MATH_UTILS_H

#include <random>
#include <memory>

class Uniform
{
public:
    Uniform(double a, double b)
    {
        std::random_device rd;
        gen_ = std::make_unique<std::mt19937>(rd());
        dis_ = std::make_unique<std::uniform_real_distribution<double>>(a, b);
    }

    double sample() const {
        return (*dis_)(*gen_);
    }

private:
    std::unique_ptr<std::mt19937> gen_;
    std::unique_ptr<std::uniform_real_distribution<double>> dis_;
};

#endif
