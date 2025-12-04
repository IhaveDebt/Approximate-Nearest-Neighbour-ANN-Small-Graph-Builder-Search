//
// HNSWLite.swift
// Small, single-file approximate nearest neighbour index inspired by HNSW ideas.
// Uses multi-layer small-world graph (simplified) with greedy search.
// Not production HNSW; educational, reasonably-sized vectors.
// Swift 5+
//

import Foundation

// Euclidean vector
typealias Vector = [Double]

func euclidean(_ a: Vector, _ b: Vector) -> Double {
    precondition(a.count == b.count)
    var s = 0.0
    for i in 0..<a.count { let d = a[i] - b[i]; s += d*d }
    return sqrt(s)
}

// Node in the graph
class HNode {
    let id: Int
    let vec: Vector
    var neighbors: [Int] = [] // neighbor ids in base layer
    var level: Int = 0 // higher-level nodes connect further (simplified)
    init(id: Int, vec: Vector, level: Int = 0) { self.id = id; self.vec = vec; self.level = level }
}

// Simplified HNSW index
class HNSWLite {
    var nodes: [Int: HNode] = [:]
    var maxLevel: Int = 0
    let m: Int // target connectivity per node
    init(m: Int = 8) { self.m = m }
    
    func add(vector: Vector) -> Int {
        let id = nodes.count
        let level = randomLevel()
        let node = HNode(id: id, vec: vector, level: level)
        nodes[id] = node
        if level > maxLevel { maxLevel = level }
        // connect into layer 0 via simple greedy: find m nearest among existing nodes
        if nodes.count > 1 {
            let neighbors = selectNeighbors(for: node, candidates: Array(nodes.values), m: m)
            node.neighbors = neighbors.map { $0.id }
            for n in node.neighbors {
                nodes[n]?.neighbors.append(node.id)
                if nodes[n]!.neighbors.count > m { nodes[n]!.neighbors.removeFirst() } // keep small
            }
        }
        return id
    }
    
    func randomLevel() -> Int {
        // simple geometric: P(level >= l) = 1/2^l
        var lvl = 0
        while Double.random(in: 0...1) < 0.5 {
            lvl += 1
        }
        return lvl
    }
    
    func selectNeighbors(for node: HNode, candidates: [HNode], m: Int) -> [HNode] {
        // compute distance to candidates, sort, pick top m (excluding itself)
        var cand = candidates.filter { $0.id != node.id }
        cand.sort { euclidean(node.vec, $0.vec) < euclidean(node.vec, $1.vec) }
        return Array(cand.prefix(m))
    }
    
    // Greedy search starting from random entry point(s)
    func knn(query: Vector, k: Int, ef: Int = 20) -> [(Int, Double)] {
        if nodes.isEmpty { return [] }
        // entry: choose a random node (or the one at maxLevel in real HNSW)
        var entry = nodes.values.first!
        var best = entry
        // greedy search on layers omitted for simplicity; do multi-start greedy on base graph
        var visited = Set<Int>()
        var candidateQueue: [(Int, Double)] = []
        func pushCandidate(_ id: Int) {
            if visited.contains(id) { return }
            visited.insert(id)
            let dist = euclidean(query, nodes[id]!.vec)
            candidateQueue.append((id, dist))
            candidateQueue.sort { $0.1 < $1.1 }
            if candidateQueue.count > ef { candidateQueue.removeLast() }
        }
        // initialize with a few random seeds
        let seeds = nodes.values.shuffled().prefix(3)
        for s in seeds { pushCandidate(s.id) }
        // expand
        var idx = 0
        while idx < candidateQueue.count {
            let (cid, _) = candidateQueue[idx]
            guard let cnode = nodes[cid] else { idx += 1; continue }
            for n in cnode.neighbors {
                pushCandidate(n)
            }
            idx += 1
        }
        // return top-k from candidateQueue
        return Array(candidateQueue.prefix(k))
    }
}

// Demo: build index on synthetic vectors and run queries
func demoHNSWLite() {
    print("=== HNSWLite Demo ===")
    let dim = 8
    let n = 2000
    let index = HNSWLite(m: 8)
    // build vectors clustered
    for i in 0..<n {
        let cluster = i % 5
        let center = (0..<dim).map { _ in Double(cluster) * 5.0 }
        let v = (0..<dim).map { j in center[j] + Double.random(in: -1...1) }
        index.add(vector: v)
    }
    // query: random point near cluster 2
    let q = (0..<dim).map { _ in Double(2) * 5.0 + Double.random(in: -0.6...0.6) }
    let results = index.knn(query: q, k: 5, ef: 50)
    print("Top-5 results (id, dist):")
    for r in results { print(" ", r.0, String(format: "%.3f", r.1)) }
}

demoHNSWLite()
