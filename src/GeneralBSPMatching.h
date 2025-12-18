#ifndef GENERALBSPMATCHING_H
#define GENERALBSPMATCHING_H

#include "BSPOT.h"
#include "coupling.h"
#include "cuda_backend.h"
#include "sampling.h"
#include "telemetry.h"
#include <random>

namespace BSPOT {

template<int D>
class GeneralBSPMatching {
public:
protected:
    using Pts = Points<D>;

    int dim;

    const Pts& A;
    const Pts& B;

    Atoms mu,nu;
    Atoms src_mu;
    Atoms src_nu;

    struct CDFSplit {
        int id;
        scalar rho;
    };

    std::vector<triplet> triplets;

    struct atom_split {
        int id = -1;
        scalar mass_left,mass_right;
    };

    Pts Grad;
    scalar W = 0;
    bool random_pivot = true;
    Coupling coupling;

    struct SliceView {
        const Atoms& id;
        int b,e;

        int operator[](int i) const {return id[b + i].id;}

        int size() const {return e - b;}
    };


public:

    GeneralBSPMatching(const Pts& A_,const Atoms& MU,const Pts& B_,const Atoms& NU) : src_mu(MU),src_nu(NU),A(A_),B(B_) {
        dim = A.rows();
        if (D != -1 && dim != D) {
            spdlog::error("dynamic dimension is different from static one !");
        }
        mu.resize(MU.size());
        nu.resize(NU.size());
        Grad = Pts::Zero(dim,MU.size());
        coupling = Coupling(mu.size(),nu.size());
    }

    GeneralBSPMatching(const Pts& A_,const Pts& B_,bool random_pivot = true) : A(A_),B(B_) {
        dim = A.rows();
        if (D != -1 && dim != D) {
            spdlog::error("dynamic dimension is different from static one !");
        }
        Grad = Pts::Zero(dim,A.cols());
        coupling = Coupling(A.cols(),B.cols());
    }

protected:


    // Partition atoms by dot value to build a split for the mass CDF.
    CDFSplit partition(Atoms &atoms, int beg, int end, int idx) {
        scalar d = atoms[idx].dot;
        int idmin = beg;
        int idmax = end-1;
        scalar sum_min = 0;
        while (idmin < idmax) {
            while (idmin < end && atoms[idmin].dot < d){
                sum_min += atoms[idmin].mass;
                idmin++;
            }
            while (idmax >= beg && atoms[idmax].dot > d)
                idmax--;
            if (idmin >= idmax)
                break;
            if (idmin < idmax)
                std::swap(atoms[idmin],atoms[idmax]);
        }
        return {idmin,sum_min};
    }

    CDFSplit quickCDF(Atoms &atoms, int beg, int end, scalar rho, scalar sum) {
        if (end - beg == 1)
            return {beg,sum};
        int idx = getRandomPivot(beg,end-1);
        auto [p,sum_min] = partition(atoms,beg,end,idx);
        if (sum_min >= rho){
            return quickCDF(atoms,beg,p,rho,sum);
        }
        else
            return quickCDF(atoms,p,end,rho - sum_min,sum + sum_min);
    }

    CDFSplit quickCDF(Atoms &atoms, int beg, int end, scalar rho) {
        return quickCDF(atoms,beg,end,rho,0);
    }

    int dotMedian(const Atoms &atoms, int a, int b, int c) {
        const auto& da = atoms[a].dot;
        const auto& db = atoms[b].dot;
        const auto& dc = atoms[c].dot;
        if ((da >= db && da <= dc) || (da >= dc && da <= db)) return a;
        if ((db >= da && db <= dc) || (db >= dc && db <= da)) return b;
        return c;
    }

    CDFSplit partitionCDF(Atoms &atoms, int beg, int end) {
        if (end - beg == 2) {
            if (atoms[beg].dot > atoms[beg+1].dot)
                std::swap(atoms[beg],atoms[beg+1]);
            return {beg+1,atoms[beg].mass};
        }
        int rand_piv = getRandomPivot(beg+1,end-2);
        int piv = dotMedian(atoms,rand_piv,beg,end-1);
        //spdlog::info("start partition b{} p{} e{}",beg,piv,end);
        return partition(atoms,beg,end,piv);
    }

    atom_split splitCDF(Atoms &atoms, int beg, int end, scalar rho) {
        auto selected = quickCDF(atoms,beg,end,rho);
        scalar mass_left = rho - selected.rho;
        scalar mass_right = atoms[selected.id].mass - mass_left;

        return {selected.id,mass_left,mass_right};
    }

    void computeDots(Atoms &atoms, const Pts &X, int beg, int end, const Vector<D> &d) {
        const int count = end - beg;
        if (count <= 0) return;
        // Jitter avoids degenerate equal projections during partitioning.
        constexpr scalar kDotJitter = static_cast<scalar>(1e-8);
        telemetry::Recorder* rec = telemetry::current();

        constexpr int kMinGpuPoints = 4096;
        if (count >= kMinGpuPoints && cuda_backend::enabled()) {
            thread_local std::vector<int> ids;
            thread_local std::vector<scalar> dots;
            ids.resize(static_cast<std::size_t>(count));
            dots.resize(static_cast<std::size_t>(count));
            for (int i = 0; i < count; ++i) {
                ids[static_cast<std::size_t>(i)] = atoms[beg + i].id;
            }
            if (cuda_backend::projectDots(X.data(), dim, X.cols(), ids.data(), count, d.data(), dots.data(), rec)) {
                for (int i = 0; i < count; ++i) {
                    const int idx = beg + i;
                    atoms[idx].dot = dots[static_cast<std::size_t>(i)] + kDotJitter * static_cast<scalar>(idx);
                }
                return;
            }
        }

        const auto start = rec ? Time::now() : TimeStamp{};
        for (auto i : range(beg, end)) {
            atoms[i].dot = X.col(atoms[i].id).dot(d) + kDotJitter * static_cast<scalar>(i);
        }
        if (rec) {
            rec->addCPUProjection(static_cast<std::size_t>(count), 1000.0 * TimeFrom(start));
        }
    }

    CovType<D> slice_basis;
    bool slice_basis_computed = false;

    Vector<D> getSlice(const Atoms &m, int begA, int endA, const Atoms &n, int begB, int endB,int h) const
    {
        if (slice_basis_computed)
            return slice_basis.col(h % dim);
        if (endA - begA < 50 || endB - begB < 50)
            return sampleUnitGaussian<D>(dim);
        return sampleUnitGaussian<D>(dim);
        CovType<D> CovA = Cov(A,m,begA,endA);
        CovType<D> CovB = Cov(B,n,begB,endB);
        CovType<D> T = W2GaussianTransportMap(CovA,CovB);
        Eigen::SelfAdjointEigenSolver<Mat> solver(T);
        return solver.eigenvectors().col(getRandomPivot(0,T.cols()-1));
    }

    int getRandomPivot(int beg, int end) const {
        if (beg == end)
            return beg;
        if (end < beg)
            spdlog::error("invalid pivot range");
        static thread_local std::random_device rd;
        static thread_local std::mt19937 rng(rd());
        std::uniform_int_distribution<int> gen(beg, end); // uniform, unbiased
        return gen(rng);
    }

    bool checkMassLeak(int begA, int endA, int begB, int endB) const {
        scalar sumA = 0,sumB = 0;
        for (auto i : range(begA,endA))
            sumA += mu[i].mass;
        for (auto i : range(begB,endB))
            sumB += nu[i].mass;
        if (std::abs(sumA - sumB) > 1e-8){
            spdlog::error("mass leak detected : sumA = {}, sumB = {}",sumA,sumB);
            return true;
        }
        return false;
    }

    void partialBSPOT(int begA, int endA, int begB, int endB,int height = 0) {
        int gapA = endA - begA;
        int gapB = endB - begB;

        if (gapA == 0 || gapB == 0){
            spdlog::error("null gap");
            return;
        }

        //        checkMassLeak(begA,endA,begB,endB);


        if (gapA == 1) {
            // Leaf case: all remaining mass on A is shipped to the B slice.
            for (auto i : range(begB,endB)) {
                if (nu[i].mass < 1e-12)
                    continue;
                Grad.col(mu[begA].id) += (B.col(nu[i].id) - A.col(mu[begA].id))*nu[i].mass;
                triplet t = {mu[begA].id,nu[i].id,nu[i].mass};
                triplets.push_back(t);
            }
            return;
        }
        if (gapB == 1) {
            // Symmetric leaf case for a single B atom.
            for (auto i : range(begA,endA)) {
                if (mu[i].mass < 1e-12)
                    continue;
                Grad.col(mu[i].id) += (B.col(nu[begB].id) - A.col(mu[i].id))*mu[i].mass;
                triplet t = {mu[i].id,nu[begB].id,mu[i].mass};
                triplets.push_back(t);
            }
            return;
        }
        const Vector<D> d = getSlice(mu,begA,endA,nu,begB,endB,height);

        computeDots(mu,A,begA,endA,d);
        computeDots(nu,B,begB,endB,d);

        CDFSplit CDFS;
        if (random_pivot) {
            CDFS = partitionCDF(mu,begA,endA);
        }
        else {
            scalar sumA = 0;
            for (auto i : range(begA,endA))
                sumA += mu[i].mass;
            CDFS = quickCDF(mu,begA,endA,0.5*sumA);
            if (CDFS.id == begA) {
                CDFS.rho = mu[CDFS.id].mass;
                CDFS.id++;
            }
        }
        int p = CDFS.id;
        scalar rho = CDFS.rho;
        auto split = splitCDF(nu,begB,endB,rho);
        int splitted_atom = nu[split.id].id;

        nu[split.id].mass = split.mass_left;
        partialBSPOT(begA,p,begB,split.id+1,height + 1);

        nu[split.id].id = splitted_atom;
        nu[split.id].mass = split.mass_right;
        partialBSPOT(p,endA,split.id,endB,height + 1);
    }

    void init() {
        for (auto i : range(src_mu.size()))
            mu[i] = src_mu[i];
        for (auto i : range(src_nu.size()))
            nu[i] = src_nu[i];
        Grad = Pts::Zero(dim,A.cols());
        triplets.clear();
        // Heuristic reserve to reduce allocations when building the sparse plan.
        triplets.reserve(static_cast<std::size_t>(src_mu.size() + src_nu.size()));
        coupling.setZero();
    }

    void setMeasures(const Atoms &mu_, const Atoms &nu_)
    {
        src_mu = mu_;
        src_nu = nu_;
        mu.resize(mu_.size());
        nu.resize(nu_.size());
    }

    Moments<D> computeMoments(const Pts& X,const Atoms& id,int b,int e) const {
        Vec masses(e-b);
        scalar S = 0;
        for (auto i : range(b,e)){
            masses(i) = id[i].mass;
            S += id[i].mass;
        }
        Eigen::DiagonalMatrix<scalar,-1> M = (masses/S).asDiagonal();
        SliceView view(id,b,e);
        Pts sub = X(Eigen::all,view);
        Pts wsub = sub*M;
        Vector<D> mean = wsub.rowwise().sum();
        Pts centered = sub.colwise() - mean;
        CovType<D> rslt = (centered*M) * centered.adjoint() / double(e-b);
        return {mean,rslt};

    }

    Vector<D> getMean(const Pts &X, const Atoms &id, int b, int e) const
    {
        Vector<D> m = Vector<D>::Zero(dim);
        scalar s = 0;
        for (auto i : range(b,e)) {
            m += X.col(id[i].id)*id[i].mass;
            s += id[i].mass;
        }
        return m/s;
    }

    CovType<D> Cov(const Pts &X, const Atoms &atoms, int b, int e) const
    {
        Vector<D> m = getMean(X,atoms,b,e);
        CovType<D> Cov = CovType<D>::Zero(dim,dim);
        scalar s = 0;
        for (auto i : range(b,e)) {
            Vector<D> x = X.col(atoms[i].id) - m;
            Cov.noalias() += x*x.transpose()*atoms[i].mass;
            s += atoms[i].mass;
        }
        return Cov/s;
    }

public:

    const Coupling &computeCoupling(bool rp = true){
        init();
        random_pivot = rp;
        if (checkMassLeak(0,src_mu.size(),0,src_nu.size())) {
            spdlog::error("cannot compute plan to unbalanced marginals");
        }
        partialBSPOT(0,src_mu.size(),0,src_nu.size());
        coupling.setFromTriplets(triplets.begin(),triplets.end());
        //coupling.makeCompressed();
        return coupling;
    }

    const Coupling &computeOrthogonalCoupling(const CovType<D>& slice_basis = CovType<D>::Identity(D,D)){
        this->slice_basis = slice_basis;
        slice_basis_computed = true;
        return computeCoupling(false);
    }


    const Pts &computeTransportGradient(bool random_pivot = true){
        init();
        this->random_pivot = random_pivot;
        partialBSPOT(0,src_mu.size(),0,src_nu.size());
        for (auto i : range(src_mu.size()))
            Grad.col(i) /= src_mu[i].mass;
        return Grad;
    }

    const Pts &computeOrthogonalTransportGradient(const CovType<D>& slice_basis = CovType<D>::Identity(D,D),bool rp = false){
        this->slice_basis = slice_basis;
        slice_basis_computed = true;
        return computeTransportGradient(rp);
    }
};

}

#endif // GENERALBSPMATCHING_H
