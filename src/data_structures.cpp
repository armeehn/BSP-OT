#include "data_structures.h"



BSPOT::UnionFind::UnionFind(int n) {
    parent.resize(n);
    rank.resize(n, 0);
    componentSize.resize(n, 1); // Initialize each component size to 1
    for (int i = 0; i < n; ++i) parent[i] = i;
}

int BSPOT::UnionFind::find(int u) {
    if (parent[u] != u) {
        parent[u] = find(parent[u]); // Path compression
    }
    return parent[u];
}

//void UnionFind::unite(int u, int v) {
//    int rootU = find(u);
//    int rootV = find(v);
//    if (rootU != rootV) {
//        if (rank[rootU] > rank[rootV]) {
//            parent[rootV] = rootU;
//        } else if (rank[rootU] < rank[rootV]) {
//            parent[rootU] = rootV;
//        } else {
//            parent[rootV] = rootU;
//            rank[rootU]++;
//        }
//    }
//}


void BSPOT::UnionFind::unite(int x, int y) {
    int rootX = find(x), rootY = find(y);
    if (rootX != rootY) {
        if (rank[rootX] > rank[rootY]) {
            parent[rootY] = rootX;
            componentSize[rootX] += componentSize[rootY];
        } else if (rank[rootX] < rank[rootY]) {
            parent[rootX] = rootY;
            componentSize[rootY] += componentSize[rootX];
        } else {
            parent[rootY] = rootX;
            componentSize[rootX] += componentSize[rootY];
            rank[rootX]++;
        }
    }
}

std::vector<std::vector<int>> BSPOT::UnionFind::getConnectedComponents(int n) {
    std::unordered_map<int, int> rootIndex;  // Maps root -> index in components
    std::vector<std::vector<int>> components;

    // **Step 1: Determine component sizes and allocate memory**
    for (int i = 0; i < n; i++) {
        int root = find(i);
        if (rootIndex.find(root) == rootIndex.end()) {
            rootIndex[root] = components.size();
            components.emplace_back();
            components.back().reserve(componentSize[root]); // Preallocate!
        }
    }

    // **Step 2: Populate components without push_back overhead**
    for (int i = 0; i < n; i++) {
        int root = find(i);
        components[rootIndex[root]].push_back(i);
    }

    return components;
}

void BSPOT::StampedPriorityQueue::insert(int key, scalar priority) {
    int ts = 0;
    if (timestamp.contains(key))
        ts = timestamp[key]+1;
    timestamp[key] = ts;
    queue.push(stamped_element{priority, key, ts});
}

std::pair<int, BSPOT::scalar> BSPOT::StampedPriorityQueue::pop() {
    if (queue.empty())
        return {-1, 0};
    stamped_element e = queue.top();
    queue.pop();
    while (timestamp[e.id] != e.timestamp) {
        if (queue.empty())
            return {-1, 0};
        e = queue.top();
        queue.pop();
    }
    return {e.id, e.priority};
}

bool BSPOT::StampedPriorityQueue::empty() const {
    return queue.empty();
}
