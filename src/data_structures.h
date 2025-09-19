#ifndef DATA_STRUCTURES_H
#define DATA_STRUCTURES_H
#include <queue>
#include <vector>
#include <set>
#include <unordered_map>

#include "BSPOT.h"
#include <iostream>

namespace BSPOT {

class UnionFind {
private:
    std::vector<int> parent, rank,componentSize;
public:
    UnionFind(int n);

    int find(int u);

    void unite(int u, int v);

    std::vector<std::vector<int>> getConnectedComponents(int n);
} ;


class StampedPriorityQueue {
private:

    struct stamped_element {
        scalar priority;
        int id;
        int timestamp;
        bool operator<(const stamped_element& other) const {
            return priority < other.priority;
        }
    };
    std::priority_queue<stamped_element> queue;
    std::map<int,int> timestamp;

public:
    void insert(int key, scalar priority);

    std::pair<int, scalar> pop();

    bool empty() const;
};


class StopWatch {
     std::map<std::string,scalar> profiler;
     TimeStamp clock;

public:
     void start() {
        clock = Time::now();
    }

     void reset() {
        profiler.clear();
    }

     void tick(std::string label) {
        if (profiler.find(label) == profiler.end())
            profiler[label] = 0;
        profiler[label] += TimeFrom(clock);
        clock = Time::now();
    }

     void profile(bool relative = true) {
        std::cout << "         STOPWATCH REPORT            " << std::endl;
        scalar s = 0;
        std::vector<std::pair<std::string,scalar>> stamps;
        for (const auto& [key,value] : profiler){
            s += value;
            stamps.push_back({key,value});
        }
        if (!relative)
            s = 1;
        std::sort(stamps.begin(),stamps.end(),[](std::pair<std::string,scalar> a,std::pair<std::string,scalar> b) {
            return a.second > b.second;
        });
        for (auto x : stamps){
            std::cout << x.first << " : " << x.second/s << "\n";
        }
        std::cout << "         END     REPORT            " << std::endl << std::endl;
    }

};

struct Edge {
    int i, j;
    scalar w;
};


class TreeGraph {
public:
    std::vector<std::unordered_map<int, scalar>> adj; // Adjacency list with unordered maps

    TreeGraph(int n) : adj(n) {} // Constructor initializes adjacency list with 'n' vertices

    void addEdge(int u, int v, scalar w) {
        adj[u][v] = w;
        adj[v][u] = w;
    }

    void changeWeight(int u, int v, scalar w) {
        if (u >= adj.size() || v >= adj.size()) return; // Out of bounds check

        auto it = adj[u].find(v);
        if (it != adj[u].end()) {
            it->second = w;
            adj[v][u] = w; // Update the reverse edge as well
        }
    }

    void removeEdge(int u, int v) {
        if (u >= adj.size() || v >= adj.size()) return;

        adj[u].erase(v);
        adj[v].erase(u);
    }

    std::vector<Edge> findPath(int start, int end) {
        if (start >= adj.size() || end >= adj.size()) return {}; // Out of bounds check

        std::unordered_map<int, Edge> parent; // Maps node -> (parent edge)
        std::queue<int> q;
        q.push(start);
        parent[start] = {-1, -1, 0}; // Root has no parent edge

        bool found = false;

        // BFS traversal
        while (!q.empty()) {
            int node = q.front();
            q.pop();

            if (node == end) {
                found = true;
                break; // Stop early when we reach the target
            }

            for (const auto& [neighbor, weight] : adj[node]) {
                if (parent.find(neighbor) == parent.end()) { // Not visited
                    parent[neighbor] = {node, neighbor, weight};
                    q.push(neighbor);
                }
            }
        }

        if (!found) return {}; // No path found

        // Reconstruct the path from end to start
        std::vector<Edge> path;
        int current = end;
        while (parent[current].i != -1) { // -1 means root node
            path.push_back(parent[current]);
            current = parent[current].i;
        }

        std::reverse(path.begin(), path.end()); // Reverse to get correct order
        return path;
    }
};

}


#endif // DATA_STRUCTURES_H
