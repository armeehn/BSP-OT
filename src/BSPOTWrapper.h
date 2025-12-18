#ifndef BSPOTWRAPPER_H
#define BSPOTWRAPPER_H

#include "BijectiveBSPMatching.h"
#include "GeneralBSPMatching.h"
#include "BijectiveMatching.h"

#include "PartialBSPMatching.h"

#include <omp.h>

namespace BSPOT {

/*
BijectiveMatching MergePlans(const std::vector<BijectiveMatching>& plans,const cost_function& cost,BijectiveMatching T = BijectiveMatching()) {
    std::vector<std::pair<scalar,int>> scores(plans.size());
#pragma omp parallel for
    for (int i = 0;i<plans.size();i++)
        scores[i] = {plans[i].evalMatching(cost),i};
    //std::sort(scores.begin(),scores.end());
    for (auto i : range(scores.size()))
        T = BSPOT::Merge(T,plans[scores[i].second],cost);
    return T;
}
*/


template<int dim>
BijectiveMatching computeGaussianBSPOT(const Points<dim>& A,const Points<dim>& B,int nb_plans,const cost_function& cost,BijectiveMatching T = BijectiveMatching()) {
    std::vector<BijectiveMatching> plans(nb_plans);
    BijectiveBSPMatching BSP(A,B);
    auto start = Time::now();
#pragma omp parallel for
    for (int i = 0;i<nb_plans;i++)
        plans[i] = BSP.computeGaussianMatching();
    return MergePlans(plans,cost,T,(A.cols() < 500000));
}

template<int dim>
BijectiveMatching computeBijectiveBSPOT(const Points<dim>& A,const Points<dim>& B,int nb_plans,const cost_function& cost,BijectiveMatching T = BijectiveMatching()) {
    std::vector<BijectiveMatching> plans(nb_plans);
    BijectiveBSPMatching BSP(A,B);
    int d = A.rows();
#pragma omp parallel for
    for (int i = 0;i<nb_plans;i++){
        plans[i] = BSP.computeMatching();
    }
    return MergePlans(plans,cost,T);
}

template<int dim>
BijectiveMatching computeBijectiveOrthogonalBSPOT(const Points<dim>& A,const Points<dim>& B,int nb_plans,const cost_function& cost,BijectiveMatching T = BijectiveMatching()) {
    std::vector<BijectiveMatching> plans(nb_plans);
    BijectiveBSPMatching BSP(A,B);
    int d = A.rows();
#pragma omp parallel for
    for (int i = 0;i<nb_plans;i++){
        Mat Q = sampleUnitGaussianMat(d,d);
        Q = Q.fullPivHouseholderQr().matrixQ();
        plans[i] = BSP.computeOrthogonalMatching(Q);
    }
    return MergePlans(plans,cost,T);
}

template<int dim>
Coupling computeBSPOTCoupling(const Points<dim>& A,const Atoms& mu,const Points<dim>& B,const Atoms& nu) {
    GeneralBSPMatching BSP(A,mu,B,nu);
    return BSP.computeCoupling();
}

template<int dim>
Points<dim> computeBSPOTGradient(const Points<dim>& A,const Atoms& mu,const Points<dim>& B,const Atoms& nu,int nb_plans) {
    Points<dim> Grad = Points<dim>::Zero(A.rows(),A.cols());
    if (nb_plans <= 0)
        return Grad;
    const int max_threads = omp_get_max_threads();
    std::vector<Points<dim>> accum(max_threads, Points<dim>::Zero(A.rows(),A.cols()));
#pragma omp parallel
    {
        const int tid = omp_get_thread_num();
#pragma omp for
        for (int i = 0; i < nb_plans; i++) {
            GeneralBSPMatching BSP(A,mu,B,nu);
            Points<dim> Grad_i = BSP.computeTransportGradient();
            accum[tid] += Grad_i;
        }
    }
    for (auto& local : accum)
        Grad += local;
    Grad /= static_cast<scalar>(nb_plans);
    return Grad;
}


template<int dim>
InjectiveMatching computePartialBSPOT(const Points<dim>& A,const Points<dim>& B,int nb_plans,const cost_function& cost,InjectiveMatching T = InjectiveMatching()) {
    std::vector<InjectiveMatching> plans(nb_plans);
    PartialBSPMatching<dim> BSP(A,B,cost);
#pragma omp parallel for
    for (int i = 0;i<nb_plans;i++){
        plans[i] = BSP.computePartialMatching();
    }
    return MergePlans(plans,cost,T);
    // InjectiveMatching plan = T;
    // for (int i = 0;i<nb_plans;i++)
    //     plan = InjectiveMatching::Merge(plan,plans[i],cost);

    // return plan;
}


template<int dim>
InjectiveMatching computePartialOrthogonalBSPOT(const Points<dim>& A,const Points<dim>& B,int nb_plans,const cost_function& cost,InjectiveMatching T = InjectiveMatching()) {
    std::vector<InjectiveMatching> plans(nb_plans);
    PartialBSPMatching<dim> BSP(A,B,cost);
#pragma omp parallel for
    for (int i = 0;i<nb_plans;i++){
        Points<dim> Q = sampleUnitGaussianMat(dim,dim).fullPivHouseholderQr().matrixQ();
        plans[i] = BSP.computePartialMatching(Q,false);
    }
    return MergePlans(plans,cost,T);
    // InjectiveMatching plan = T;
    // for (int i = 0;i<nb_plans;i++)
    //     plan = InjectiveMatching::Merge(plan,plans[i],cost);

    // return plan;
}



}

#endif // BSPOTWRAPPER_H
