#include "InjectiveMatching.h"


BSPOT::InjectiveMatching::InjectiveMatching(int m) : image_domain_size(m) {}

BSPOT::InjectiveMatching::InjectiveMatching() {}

BSPOT::InjectiveMatching::InjectiveMatching(const TransportPlan &T, int m) : image_domain_size(m),plan(T) {

}

BSPOT::scalar BSPOT::InjectiveMatching::evalMatching(const cost_function &cost) const {
    scalar c = 0;
    for (auto i : range(plan.size()))
        c += cost(i,plan[i])/plan.size();
    return c;
}

const BSPOT::InjectiveMatching::TransportPlan &BSPOT::InjectiveMatching::getPlan() const {return plan;}

size_t BSPOT::InjectiveMatching::operator[](size_t i) const {return plan[i];}

size_t BSPOT::InjectiveMatching::operator()(size_t i) const {return plan[i];}

size_t BSPOT::InjectiveMatching::size() const {return plan.size();}

BSPOT::InjectiveMatching::operator TransportPlan() const {return plan;}

bool BSPOT::InjectiveMatching::swapIfUpgrade(ints &T, ints &TI, const ints &TP, int a, const cost_function &cost) {
    int b = T[a];
    int bp = TP[a];
    int ap  = TI[bp];
    if (a == ap || b == bp)
        return false;
    if (a == ap || b == bp)
        return false;
    if (ap != -1) {
        if (cost(ap,b) + cost(a,bp) < cost(a,b) + cost(ap,bp) ){
            T[a] = bp;
            T[ap] = b;
            TI[bp] = a;
            TI[b] = ap;
            return true;
        }
    }
    else {
        if (cost(a,bp) < cost(a,b)) {
            T[a] = bp;
            TI[b] = -1;
            TI[bp] = a;
            return true;
        }
    }
    return false;
}

BSPOT::InjectiveMatching::InverseTransportPlan BSPOT::InjectiveMatching::inversePlan() {
    if (inverse_plan.empty())
        inverse_plan = getInverse();
    return inverse_plan;
}

BSPOT::InjectiveMatching::InverseTransportPlan BSPOT::InjectiveMatching::inversePlan() const {
    if (inverse_plan.empty())
        spdlog::error("inverse plan not computed");
    return inverse_plan;
}

const BSPOT::InjectiveMatching::TransportPlan &BSPOT::InjectiveMatching::getInversePlan() {
    inverse_plan = getInverse();
    return inverse_plan;
}



BSPOT::InjectiveMatching::InverseTransportPlan BSPOT::InjectiveMatching::getInverse() const {
    if (image_domain_size == -1) {
        spdlog::info("cannot compute inverse if image domain size is not filled");
        return {};
    }
    InverseTransportPlan rslt(image_domain_size,-1);
    for (auto i : range(plan.size()))
        rslt[plan[i]] = i;
    return rslt;
}

bool checkValid(const BSPOT::ints &T,const BSPOT::ints& TI) {
    int M = TI.size();
    std::set<int> image;
    for (auto i : BSPOT::range(T.size())) {
        if (T[i] == -1)
            return false;
        image.insert(T[i]);
    }
    if (image.size() != T.size()){
        spdlog::error("not injective");
        return false;
    }
    for (auto i : BSPOT::range(T.size()))
        if (TI[T[i]] != i){
            spdlog::error("wrong inverse");
            return false;
        }
    for (auto i : BSPOT::range(M)){
        if (TI[i] != -1 && !image.contains(i)){
            spdlog::error("wrong inverse");
            return false;
        }
    }
    return true;
}


BSPOT::InjectiveMatching BSPOT::InjectiveMatching::Merge(const InjectiveMatching &T, const InjectiveMatching &TP, const cost_function &cost)
{
    if (T.size() == 0)
        return TP;
    int N = T.size();
    int M = T.image_domain_size;

    UnionFind UF(N + M);
    for (auto i : range(N)) {
        UF.unite(i,T[i]+N);
        UF.unite(i,TP[i]+N);
    }

    std::map<int,ints> components;
    for (auto i  = 0;i<N;i++) {
        auto p = UF.find(i);
        components[p].push_back(i);
    }

    ints rslt = T;
    ints rsltI = T.getInverse();
    ints Tp = TP;

    std::vector<ints> connected_components(components.size());
    int i = 0;
    for (auto& [p,cc] : components)
        connected_components[i++] = cc;


#pragma omp parallel for
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
                rsltI[rslt[i]] = -1;
            for (auto i : c)
                std::swap(Tp[i],rslt[i]);
            for (auto i : c)
                rsltI[rslt[i]] = i;
        }
        for (auto i : c)
            InjectiveMatching::swapIfUpgrade(rslt,rsltI,Tp,i,cost);
    }
    //checkValid(rslt,rsltI);
    return InjectiveMatching(rslt,M);
}

BSPOT::Vec evalMappings(const BSPOT::InjectiveMatching& T,const BSPOT::cost_function& cost) {
    BSPOT::Vec costs(T.size());
    for (int i = 0;i<T.size();i++)
        costs[i] = cost(i,T[i]);

    return costs;
}


BSPOT::InjectiveMatching BSPOT::MergePlans(const std::vector<InjectiveMatching> &plans, const cost_function &cost, BSPOT::InjectiveMatching T) {
    int s = 0;
    if (T.size() == 0) {
        T = plans[0];
        s = 1;
    }
    int N = plans[0].size();

    auto C = evalMappings(T,cost);

    ints rslt = T;
    ints rsltI = T.getInverse();

    for (auto k : range(s,plans.size())) {

        auto Cp = evalMappings(plans[k],cost);

        const auto& Tp = plans[k];
        for (auto a : range(N))
        {
            int b = rslt[a];
            int bp = Tp[a];
            int ap  = rsltI[bp];
            if (a == ap || b == bp)
                continue;
            if (ap != -1) {
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
            } else {
                scalar old_cost = C[a];
                scalar cabp = cost(a,bp);
                if (cabp < old_cost) {
                    rslt[a] = bp;
                    rsltI[b] = -1;
                    rsltI[bp] = a;
                }
            }
        }
    }
    return InjectiveMatching(rslt,plans[0].image_domain_size);
}
