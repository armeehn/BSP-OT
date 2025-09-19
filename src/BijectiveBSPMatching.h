#ifndef BIJECTIVEBSPMATCHING_H
#define BIJECTIVEBSPMATCHING_H

#include "BSPOT.h"
#include "spdlog/spdlog.h"
#include "BijectiveMatching.h"
#include "../common/sampling.h"

namespace BSPOT {

template<int D>
class BijectiveBSPMatching {
public:
    using TransportPlan = ints;

    using Pts = BSPOT::Points<D>;
    const Pts& A;
    const Pts& B;
    int dim;

protected:

    struct dot_id {
        scalar dot;
        int id;
        bool operator<(const dot_id& other) const {
            return dot < other.dot;
        }
    };

    using ids = std::vector<dot_id>;
    struct SliceView {
        const ids& id;
        int b,e;

        int operator[](int i) const {return id[b + i].id;}

        int size() const {return e - b;}
    };


    static Moments<D> computeMoments(const Pts& mat,const ids& I,int b,int e) {
        SliceView view(I,b,e);
        thread_local static Pts sub;
        sub = mat(Eigen::all,view);
        Vector<D> mean = sub.rowwise().mean();

        CovType<D> rslt = CovType<D>::Zero(mat.rows(),mat.rows());
        for (auto i : range(sub.cols())){
            Vector<D> c = sub.col(i) - mean;
            rslt += c*c.transpose()/scalar(e-b);
        }
        // Pts centered = sub.colwise() - mean;
        // CovType<D> rslt = centered * centered.adjoint() / double(e-b);
        return {mean,rslt};
    }



    Vector<D> getSlice(ids &idA,ids &idB, int b, int e) {
        return sampleUnitGaussian<D>(dim);
    }

    void BSP(ids& idA,ids& idB,int beg,int end,int pivot,const Vector<D>& d) {

        for (auto i : range(beg,end)) {
            idA[i].dot = d.dot(A.col(idA[i].id));// + sampleUnitGaussian<1>()(0)*0e-3;
            idB[i].dot = d.dot(B.col(idB[i].id));// + sampleUnitGaussian<1>()(0)*0e-3;
        }
        std::nth_element(idA.begin() + beg,idA.begin() + pivot,idA.begin() + end);
        std::nth_element(idB.begin() + beg,idB.begin() + pivot,idB.begin() + end);
    }


    bool random_pivot = true;

    std::pair<Moments<D>,Moments<D>> decomposeMoments(const Pts& X,const Moments<D>& M, const ids& id, int beg, int end,int pivot) {
        scalar alpha = scalar(pivot - beg)/scalar(end - beg);
        scalar beta = 1 - alpha;

        auto [ML,CL] = computeMoments(X,id,beg,pivot);

        Vector<D> MR = (M.mean - alpha*ML)/beta;
        CovType<D> DL = (M.mean - ML)*(M.mean - ML).transpose();
        CovType<D> DR = (M.mean - MR)*(M.mean - MR).transpose();
        CovType<D> CR = CovType<D>(M.Cov - alpha*(CL + DL))/beta - DR;

        return {{ML,CL},{MR,CR}};
    }

    bool init_mode = false;

    Vector<D> DrawEigenVector(const CovType<D> &GT) {
        Eigen::SelfAdjointEigenSolver<CovType<D>> solver(GT);
        return solver.eigenvectors().col(randint(0,dim-1));
    }


    Vector<D> gaussianSlice(const Moments<D>& MA,const Moments<D>& MB) {
        CovType<D> GT = W2GaussianTransportMap(MA.Cov,MB.Cov);
        return DrawEigenVector(GT);
    }


    void gaussianPartialBSPOT(ids &idA, ids &idB, int beg, int end, const Moments<D>& MA,const Moments<D>& MB) {
        auto gap = (end-beg);
        if (gap == 0){
            spdlog::error("end - beg == 0");
            return;
        }
        if (gap == 1)
            return;
        if (gap < 50) {
            // random_pivot = true;
            // partialBSPOT(idA,idB,beg,end);
            partialOrthogonalBSPOT(idA,idB,beg,end,sampleUnitGaussian<D>(dim));
            // random_pivot = false;
            return;
        }

        const Vector<D> d = gaussianSlice(MA,MB);


        // int pivot = randint(beg + gap/4,beg + gap*3/4);
        int pivot = random_pivot ? randint(beg+1,end-1) : beg + (end-beg)/2;

        // for (auto i : range(beg,end)) {
        //     idA[i].dot = d.dot(A.col(idA[i].id));
        //     idB[i].dot = d.dot(B.col(idB[i].id));
        // }
        // std::nth_element(idA.begin() + beg,idA.begin() + pivot,idA.begin() + end);
        // std::nth_element(idB.begin() + beg,idB.begin() + pivot,idB.begin() + end);
        BSP(idA,idB,beg,end,pivot,d);

        auto SMA = decomposeMoments(A,MA,idA,beg,end,pivot);
        auto SMB = decomposeMoments(B,MB,idB,beg,end,pivot);

        gaussianPartialBSPOT(idA,idB,beg,pivot,SMA.first,SMB.first);
        gaussianPartialBSPOT(idA,idB,pivot,end,SMA.second,SMB.second);
    }

    Mat sliceBasis;
    bool hasSliceBasis = false;

    void partialBSPOT(ids &idA, ids &idB, int beg, int end,int height = 0) {
        auto gap = (end-beg);
        if (gap == 0){
            spdlog::error("end - beg == 0");
        }
        if (gap == 1){
            return;
        }
        int pivot = random_pivot ? randint(beg+1,end-1) : beg + (end-beg)/2;
        const Vector<D> d = hasSliceBasis ? sliceBasis.col(height % dim) : getSlice(idA,idB,beg,end);
        BSP(idA,idB,beg,end,pivot,d);
        partialBSPOT(idA,idB,beg,pivot,height+1);
        partialBSPOT(idA,idB,pivot,end,height+1);
    }

    void selectBSPOT(std::map<int,int>& T,ids &idA, ids &idB, int beg, int end,std::set<int> targets,int height = 0) {
        auto gap = (end-beg);
        if (gap == 0){
            spdlog::error("end - beg == 0");
        }
        if (gap == 1){
            if (!targets.contains(idA[beg].id))
                spdlog::error("target not found");
            T[idA[beg].id] = idB[beg].id;
            return;
        }
        int pivot = random_pivot ? randint(beg+1,end-1) : beg + (end-beg)/2;
        const Vector<D> d = hasSliceBasis ? sliceBasis.col(height % dim) : getSlice(idA,idB,beg,end);
        BSP(idA,idB,beg,end,pivot,d);
        std::set<int> L,R;
        for (auto i : range(beg,pivot))
            if (targets.contains(idA[i].id))
                L.insert(idA[i].id);
        for (auto i : range(pivot,end))
            if (targets.contains(idA[i].id))
                R.insert(idA[i].id);
        if (L.size())
            selectBSPOT(T,idA,idB,beg,pivot,L,height+1);
        if (R.size())
            selectBSPOT(T,idA,idB,pivot,end,R,height+1);
    }



    void partialOrthogonalBSPOT(ids &idA, ids &idB, int beg, int end,Vector<D> prev_slice) {
        auto gap = (end-beg);
        if (gap == 0){
            spdlog::error("end - beg == 0");
            //return;
        }
        if (gap == 1){
            return;
        }
        int pivot = random_pivot ? randint(beg+1,end-1) : beg + (end-beg)/2;
        Vector<D> d = getSlice(idA,idB,beg,end);
        d -= d.dot(prev_slice)*prev_slice/prev_slice.squaredNorm();
        d.normalized();
        BSP(idA,idB,beg,end,pivot,d);
        partialOrthogonalBSPOT(idA,idB,beg,pivot,d);
        partialOrthogonalBSPOT(idA,idB,pivot,end,d);
    }



public:

    BijectiveBSPMatching(const Pts& A_,const Pts& B_) : A(A_),B(B_) {
        dim = A.rows();
        if (D != -1 && dim != D) {
            spdlog::error("dynamic dimension is different from static one !");
        }
    }

    std::map<int,int> quickselectTransport(const std::set<int>& targets,const Mat& _sliceBasis) {
        sliceBasis = _sliceBasis;
        hasSliceBasis = true;
        return quickselectTransport(targets);
    }

    std::map<int,int> quickselectTransport(const std::set<int>& targets) {
        ids idA(A.cols()),idB(B.cols());
        for (auto i : range(A.cols())) {
            idA[i].id = i;
            idB[i].id = i;
        }
        std::map<int,int> T;
        selectBSPOT(T,idA,idB,0,A.cols(),targets);
        return T;
    }


    BijectiveMatching computeMatching(bool random_pivot = true){
        ids idA(A.cols()),idB(B.cols());
        for (auto i : range(A.cols())) {
            idA[i].id = i;
            idB[i].id = i;
        }

        this->random_pivot = random_pivot;
        partialBSPOT(idA,idB,0,A.cols());

        ints plan = TransportPlan(A.cols());
        for (int i = 0;i<A.cols();i++)
            plan[idA[i].id] = idB[i].id;
        return BijectiveMatching(plan);
    }

    BijectiveMatching computeOrthogonalMatching(const Mat& _sliceBasis,bool random_pivot_ = true){
        ids idA(A.cols()),idB(B.cols());
        for (auto i : range(A.cols())) {
            idA[i].id = i;
            idB[i].id = i;
        }

        hasSliceBasis = true;
        sliceBasis = _sliceBasis;

        this->random_pivot = random_pivot_;
        partialBSPOT(idA,idB,0,A.cols());

        ints plan = TransportPlan(A.cols());
        for (int i = 0;i<A.cols();i++)
            plan[idA[i].id] = idB[i].id;
        return BijectiveMatching(plan);
    }


    BijectiveMatching computeGaussianMatching(){
        ids idA(A.cols()),idB(B.cols());
        for (auto i : range(A.cols())) {
            idA[i].id = i;
            idB[i].id = i;
        }

        random_pivot = false;

        Vector<D> meanA = A.rowwise().mean();
        Vector<D> meanB = B.rowwise().mean();
        Moments<D> MA = {meanA,Covariance(A)};
        Moments<D> MB = {meanB,Covariance(B)};

        gaussianPartialBSPOT(idA,idB,0,A.cols(),MA,MB);


        ints plan = TransportPlan(A.cols());
        for (int i = 0;i<A.cols();i++)
            plan[idA[i].id] = idB[i].id;
        return BijectiveMatching(plan);
    }

    std::pair<ints,ints> computeGaussianMatchingOrders(){
        ids idA(A.cols()),idB(B.cols());
        for (auto i : range(A.cols())) {
            idA[i].id = i;
            idB[i].id = i;
        }

        random_pivot = false;

        Vector<D> meanA = A.rowwise().mean();
        Vector<D> meanB = B.rowwise().mean();
        Moments<D> MA = {meanA,Covariance(A)};
        Moments<D> MB = {meanB,Covariance(B)};



        // partialBSPOT(idA,idB,0,A.cols());
        // partialOrthogonalBSPOT(idA,idB,0,A.cols(),sampleUnitGaussian<D>(dim));
        gaussianPartialBSPOT(idA,idB,0,A.cols(),MA,MB);
        ints OA(A.cols()),OB(A.cols());
        for (int i = 0;i<A.cols();i++)
            OA[i] = idA[i].id;
        for (int i = 0;i<A.cols();i++)
            OB[i] = idB[i].id;
        return {OA,OB};
    }


};

}

#endif // BIJECTIVEBSPMATCHING_H
