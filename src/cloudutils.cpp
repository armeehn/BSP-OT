#include "cloudutils.h"
#include <random>

/*
namespace BSPOT {

void normalize(Vecs &X, Vec offset, scalar dilat){
    int dim = X[0].size();
    Vec min = Vec::Ones(dim)*1e9;
    Vec max = Vec::Ones(dim)*(-1e9);
    for (const auto& x : X){
        min = min.cwiseMin(x);
        max = max.cwiseMax(x);
    }
    Vec scale = max - min;
    double f = dilat/scale.maxCoeff();
    if (!offset.size())
        offset = Vec::Zero(dim);
    Vec c = (min+max)*0.5;
    for (auto& x : X){
        x = (x-c)*f + offset;
    }
}

Vecs concat(const Vecs &X, const Vecs &Y)
{
    Vecs rslt(X.begin(),X.end());
    rslt.insert(rslt.end(),Y.begin(),Y.end());
    return rslt;
}

Vecs pad(const Vecs &X, int target) {
    int n = X.size();
    Vecs rslt = X;
    while (rslt.size() != target)
        rslt.push_back(X[rand()%X.size()]);
    return rslt;
}

Vecs trunc(const Vecs &X, int target){
    Vecs rslt = X;
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(rslt.begin(),rslt.end(),g);
    rslt.resize(target);
    return rslt;
}

void translate(Vecs &X, Vec offset)
{
    for (auto& x : X)
        x += offset;
}

void normalize(Mat &X, Vec offset, scalar dilat)
{
    Vec min = X.colwise().minCoeff();
    Vec max = X.colwise().maxCoeff();
    Vec scale = max - min;
    double f = dilat/scale.maxCoeff();
    if (!offset.size())
        offset = Vec::Zero(X.cols());
    Vec c = (min+max)*0.5;
    X.rowwise() -= c.transpose();
    X *= f;
    X.rowwise() += offset.transpose();

    // for (auto i : range(X.rows()))
        // X.row(i) = (X.row(i)-c).array()*f + offset.array();
}

}

*/
