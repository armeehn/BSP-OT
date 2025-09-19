#include "BijectiveMatching.h"

namespace BSPOT {

BijectiveMatching::BijectiveMatching(){}

BijectiveMatching::BijectiveMatching(const Eigen::Vector<int, -1> &T) {
    plan.resize(T.size());
    for (auto i : range(T.size()))
        plan[i] = T[i];
    inverse_plan = getInverse(plan);
}

scalar BijectiveMatching::evalMatching(const cost_function &cost) const {
    scalar c = 0;
    if (plan.empty()) {
        spdlog::error("tried to eval cost on empty plan!");
        return 0;
    }

    for (auto i : range(plan.size()))
        c += cost(i,plan.at(i));
    return c/plan.size();
}

const BijectiveMatching::TransportPlan &BijectiveMatching::getPlan() const {return plan;}

size_t BijectiveMatching::operator[](size_t i) const {return plan.at(i);}

size_t BijectiveMatching::operator()(size_t i) const {return plan.at(i);}

size_t BijectiveMatching::size() const {return plan.size();}

BijectiveMatching::operator TransportPlan() const {
    return plan;
}

BijectiveMatching BijectiveMatching::inverseMatching() {
    if (inverse_plan.empty())
        inverse_plan = getInversePlan();
    return BijectiveMatching(inverse_plan,plan);
}

BijectiveMatching BijectiveMatching::inverseMatching() const {
    if (inverse_plan.empty())
        return BijectiveMatching(getInverse(plan),plan);
    return BijectiveMatching(inverse_plan,plan);
}

bool BijectiveMatching::checkBijectivity() const
{
    auto I = getInverse(plan);
    for (auto i : I)
        if (i == -1)
            return false;
    return true;
}

BijectiveMatching BijectiveMatching::operator()(const BijectiveMatching &other) const {
    TransportPlan rslt(other.size());
    for (auto i : range(other.size()))
        rslt[i] = plan[other[i]];
    return rslt;
}

BijectiveMatching::BijectiveMatching(const TransportPlan &T, const TransportPlan &TI) : plan(T),inverse_plan(TI) {}

const BijectiveMatching::TransportPlan &BijectiveMatching::getInversePlan() {
    if (inverse_plan.empty())
        inverse_plan = getInverse(plan);
    return inverse_plan;
}

BijectiveMatching::TransportPlan BijectiveMatching::getInverse(const TransportPlan &T) {
    TransportPlan TI(T.size(),-1);
    for (auto i : range(T.size())){
        TI[T[i]] = i;
    }
    return TI;
}

template<class T>
std::vector<T> BijectiveMatching::operator()(const std::vector<T> &X) {
    std::vector<T> rslt(X.size());
    for (auto i : range(X.size()))
        rslt[plan[i]] = X[i];
    return rslt;
}


BijectiveMatching Merge(const BijectiveMatching &T, const BijectiveMatching &TP, const cost_function &cost, bool verbose) {
    if (T.size() == 0)
        return TP;
    int N = T.size();

    UnionFind UF(N*2);
    for (auto i : range(N)) {
        UF.unite(i,T[i]+N);
        UF.unite(i,TP[i]+N);
    }

    std::unordered_map<int,ints> components;
    for (auto i  = 0;i<N;i++) {
        auto p = UF.find(i);
        components[p].push_back(i);
    }

    BijectiveMatching::TransportPlan rslt = T;
    BijectiveMatching::TransportPlan rsltI = T.inverseMatching();
    BijectiveMatching::TransportPlan Tp = TP;

    std::vector<ints> connected_components(components.size());
    int i = 0;
    for (auto& [p,cc] : components)
        connected_components[i++] = cc;


    for (int k = 0;k<connected_components.size();k++) {
        const auto& c = connected_components[k];

        if (c.size() == 1)
            continue;
        scalar costT = 0,costTP = 0;
        for (auto i : c) {
            costT  += cost(i,T[i]);
            costTP += cost(i,TP[i]);
        }
        if (costTP < costT){
            for (auto i : c)
                std::swap(Tp[i],rslt[i]);
            for (auto i : c)
                rsltI[rslt[i]] = i;
        }
        for (auto i : c)
            BijectiveMatching::swapIfUpgrade(rslt,rsltI,Tp,i,cost);
    }
    return rslt;
}

Vec evalMappings(const BijectiveMatching& T,const cost_function& cost) {
//    return (A - B(Eigen::all,T)).colwise().squaredNorm();
    Vec costs(T.size());
//#pragma omp parallel for
    for (int i = 0;i<T.size();i++) {
        costs[i] = cost(i,T[i]);
    }
    return costs;
}

BijectiveMatching MergePlans(const std::vector<BijectiveMatching> &plans, const cost_function &cost, BijectiveMatching T,bool cycle) {
    int s = 0;
    auto I = true ? rankPlans(plans,cost) : rangeVec(plans.size());
    if (T.size() == 0) {
        T = plans[I[0]];
        s = 1;
    }
    int N = plans[0].size();

    auto C = evalMappings(T,cost);

    ints rslt = T;
    ints rsltI = T.inverseMatching();

    ints sig(N);

    scalar avg_cc_size = 0;

    StopWatch profiler;
    for (auto k : range(s,plans.size())) {
        ints Tp = plans[I[k]];
        ints Tpi = plans[I[k]].inverseMatching();
        auto Cp = evalMappings(Tp,cost);

        for (auto i : range(N))
            sig[i] = Tpi[rslt[i]];

        // profiler.start();

        std::vector<ints> CCs;

        if (cycle) {
            ints visited(N,-1);
            int c = 0;
            for (auto i : range(N)) {
                if (visited[i] != -1)
                    continue;
                int j = i;
                int i0 = i;
                if (sig[j] == i)
                    continue;

                ints CC;
                scalar costT = 0;
                scalar costTP = 0;

                while (visited[j] == -1) {
                    CC.push_back(j);
                    costT  += C[j];
                    costTP += Cp[j];
                    visited[j] = c;
                    j = sig[j];
                }

                if (costTP < costT) {
                    j = i0;
                    do {
                        std::swap(Tp[j],rslt[j]);
                        std::swap(C[j],Cp[j]);
                        j = sig[j];
                    } while (j != i0);
                    j = i0;
                    do {
                        rsltI[rslt[j]] = j;
                        j = sig[j];
                    } while (j != i0);
                }

                c++;
                CCs.push_back(CC);
                avg_cc_size += CC.size();
            }
        } else {
            CCs.push_back(rangeVec(N));
        }
        // profiler.tick("cycle");
        // for (auto a : range(N))
//        spdlog::info("nb cycles {} avg size {}",CCs.size(),avg_cc_size / CCs.size() );
// #pragma omp parallel for
#pragma omp parallel
        {
#pragma omp single
            {
                for (int i = 0; i < CCs.size(); ++i) {
#pragma omp task firstprivate(i)
                    {
                        for (auto a : CCs[i]){
                            // swapIfUpgradeK(rslt,rsltI,Tp,a,3,cost);
                            int b = rslt[a];
                            int bp = Tp[a];
                            int ap  = rsltI[bp];
                            if (a == ap || b == bp)
                                continue;
                            scalar old_cost = C[a] + C[ap];
                            scalar cabp = Cp[a];
                            if (cabp > old_cost)
                                continue;
                            scalar capb = cost(ap,b);
                            if (cabp + capb < old_cost) {
                                rslt[a] = bp;
                                rslt[ap] = b;
                                rsltI[bp] = a;
                                rsltI[b] = ap;
                                C[a] = cabp;
                                C[ap] = capb;
                            }
                        }
                    }
                }
            }
        }
        // for (const auto& cc : CCs)
        // {
        //     std::cout << "cc size " << cc.size() << std::endl;
        // }
        // profiler.tick("greedy");
    }
    // profiler.profile(false);
    return rslt;
}

BijectiveMatching MergePlansNoPar(const std::vector<BijectiveMatching> &plans, const cost_function &cost, BijectiveMatching T,bool cycle) {
    int s = 0;
    auto I = true ? rankPlans(plans,cost) : rangeVec(plans.size());
    if (T.size() == 0) {
        T = plans[I[0]];
        s = 1;
    }
    int N = plans[0].size();

    auto C = evalMappings(T,cost);

    ints rslt = T;
    ints rsltI = T.inverseMatching();

    ints sig(N);

    StopWatch profiler;
    for (auto k : range(s,plans.size())) {
        ints Tp = plans[I[k]];
        ints Tpi = plans[I[k]].inverseMatching();
        auto Cp = evalMappings(Tp,cost);

        for (auto i : range(N))
            sig[i] = Tpi[rslt[i]];

        // profiler.start();

        std::vector<ints> CCs;

        if (cycle) {
            ints visited(N,-1);
            int c = 0;
            for (auto i : range(N)) {
                if (visited[i] != -1)
                    continue;
                int j = i;
                int i0 = i;
                if (sig[j] == i)
                    continue;

                ints CC;
                scalar costT = 0;
                scalar costTP = 0;

                while (visited[j] == -1) {
                    CC.push_back(j);
                    costT  += C[j];
                    costTP += Cp[j];
                    visited[j] = c;
                    j = sig[j];
                }

                if (costTP < costT) {
                    j = i0;
                    do {
                        std::swap(Tp[j],rslt[j]);
                        std::swap(C[j],Cp[j]);
                        j = sig[j];
                    } while (j != i0);
                    j = i0;
                    do {
                        rsltI[rslt[j]] = j;
                        j = sig[j];
                    } while (j != i0);
                }

                c++;
                CCs.push_back(CC);
            }
        } else {
            CCs.push_back(rangeVec(N));
        }
        for (int i = 0; i < CCs.size(); ++i) {
            {
                for (auto a : CCs[i]){
                    // swapIfUpgradeK(rslt,rsltI,Tp,a,3,cost);
                    int b = rslt[a];
                    int bp = Tp[a];
                    int ap  = rsltI[bp];
                    if (a == ap || b == bp)
                        continue;
                    scalar old_cost = C[a] + C[ap];
                    scalar cabp = Cp[a];
                    if (cabp > old_cost)
                        continue;
                    scalar capb = cost(ap,b);
                    if (cabp + capb < old_cost) {
                        rslt[a] = bp;
                        rslt[ap] = b;
                        rsltI[bp] = a;
                        rsltI[b] = ap;
                        C[a] = cabp;
                        C[ap] = capb;
                    }
                }
            }
        }
    }
    return rslt;
}

BijectiveMatching load_plan(std::string path) {
    std::ifstream file(path);
    ints plan;
    while (file) {
        int i;
        file >> i;
        plan.push_back(i);
    }
    //remove last element
    plan.pop_back();
    return plan;
}


template<class T>
inline std::vector<std::vector<T>> getPermutations(std::vector<T> C) {
    std::vector<std::vector<T>> rslt;
    do
    {
        rslt.push_back(C);
    }
    while (std::next_permutation(C.begin(), C.end()));
    return rslt;
}


bool swapIfUpgradeK(ints &plan, ints &inverse_plan, const ints &T, int a, int k, const cost_function &cost)
{
    if (k == 2) {
        return BijectiveMatching::swapIfUpgrade(plan,inverse_plan,T,a,cost);
    }
    scalar s = 0;
    std::set<int> A,TA;
    A.insert(a);
    TA.insert(plan[a]);
    auto i = a;
    for (auto k : range(k-1)) {
        auto j = T[i];
        i = inverse_plan[j];
        A.insert(i);
        TA.insert(j);
    }
    if (TA.size() != A.size() || TA.size() == 1)
        return BijectiveMatching::swapIfUpgrade(plan,inverse_plan,T,a,cost);
    ints TAvec(TA.begin(),TA.end());
    ints Avec(A.begin(),A.end());
    auto Sig = getPermutations(TAvec);
    ints best;
    scalar score = 1e8;

    scalar curr = 0;
    for (auto i : range(A.size()))
        curr += cost(Avec[i],plan[Avec[i]]);

    for (const auto& sig : Sig) {
        scalar c = 0;
        for (auto i : range(sig.size()))
            c += cost(Avec[i],sig[i]);
        if (Smin(score,c))
            best = sig;
    }
    if (score > curr)
        return false;
    for (auto i : range(best.size())){
        plan[Avec[i]] = best[i];
        inverse_plan[best[i]] = Avec[i];
    }
    return true;
}

}
