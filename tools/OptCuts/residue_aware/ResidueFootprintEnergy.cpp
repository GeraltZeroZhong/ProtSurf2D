#include "ResidueFootprintEnergy.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <numeric>
#include <set>
#include <stdexcept>
#include <unordered_set>

namespace {

class UnionFind {
public:
    explicit UnionFind(int size) : parent_(size), rank_(size, 0) {
        std::iota(parent_.begin(), parent_.end(), 0);
    }

    int find(int item) {
        int root = item;
        while(parent_[root] != root) {
            root = parent_[root];
        }
        while(parent_[item] != item) {
            const int next = parent_[item];
            parent_[item] = root;
            item = next;
        }
        return root;
    }

    void unite(int left, int right) {
        int rootLeft = find(left);
        int rootRight = find(right);
        if(rootLeft == rootRight) {
            return;
        }
        if(rank_[rootLeft] < rank_[rootRight]) {
            std::swap(rootLeft, rootRight);
        }
        parent_[rootRight] = rootLeft;
        if(rank_[rootLeft] == rank_[rootRight]) {
            ++rank_[rootLeft];
        }
    }

private:
    std::vector<int> parent_;
    std::vector<std::uint8_t> rank_;
};

std::uint64_t edgeKey(int vertex0, int vertex1) {
    const std::uint32_t low = static_cast<std::uint32_t>(std::min(vertex0, vertex1));
    const std::uint32_t high = static_cast<std::uint32_t>(std::max(vertex0, vertex1));
    return (static_cast<std::uint64_t>(low) << 32U) | high;
}

void expectTag(std::istream& input, const char* expected) {
    std::string actual;
    input >> actual;
    if(!input || actual != expected) {
        throw std::runtime_error(std::string("Expected '") + expected + "' in footprint sidecar.");
    }
}

} // namespace

namespace TopoPPI {

ResidueFootprintEnergy ResidueFootprintEnergy::load(const std::string& path) {
    std::ifstream input(path);
    if(!input) {
        throw std::runtime_error("Could not open footprint sidecar: " + path);
    }

    expectTag(input, "TOPOPPI_FOOTPRINT_V2");
    expectTag(input, "COUNTS");
    int residueCount = 0;
    int edgeCount = 0;
    int vertexCount = 0;
    ResidueFootprintEnergy result;
    input >> result.faceCount_ >> residueCount >> edgeCount >> vertexCount;
    if(result.faceCount_ < 0 || residueCount < 0 || edgeCount < 0 || vertexCount < 0) {
        throw std::runtime_error("Negative count in footprint sidecar.");
    }
    result.residues_.resize(residueCount);
    result.edges_.resize(edgeCount);
    result.cut_.assign(edgeCount, 0);

    expectTag(input, "SOURCES");
    result.inputSourceVertices_.resize(vertexCount);
    for(int vertex = 0; vertex < vertexCount; ++vertex) {
        input >> result.inputSourceVertices_[vertex];
        if(!input || result.inputSourceVertices_[vertex] < 0) {
            throw std::runtime_error("Invalid input source vertex in footprint sidecar.");
        }
    }

    expectTag(input, "WEIGHTS");
    for(int residue = 0; residue < residueCount; ++residue) {
        input >> result.residues_[residue].objectiveWeight;
        if(!input || result.residues_[residue].objectiveWeight < 0.0) {
            throw std::runtime_error("Invalid residue weight in footprint sidecar.");
        }
    }

    for(int row = 0; row < result.faceCount_; ++row) {
        expectTag(input, "FACE");
        int face = -1;
        int entryCount = 0;
        input >> face >> entryCount;
        if(face < 0 || face >= result.faceCount_ || entryCount < 0) {
            throw std::runtime_error("Invalid FACE record in footprint sidecar.");
        }
        for(int entry = 0; entry < entryCount; ++entry) {
            int residue = -1;
            double mass = 0.0;
            input >> residue >> mass;
            if(residue < 0 || residue >= residueCount || !std::isfinite(mass) || mass < 0.0) {
                throw std::runtime_error("Invalid FACE mass in footprint sidecar.");
            }
            Residue& footprint = result.residues_[residue];
            footprint.faceToLocal[face] = static_cast<int>(footprint.faces.size());
            footprint.faces.push_back(face);
            footprint.masses.push_back(mass);
            footprint.totalMass += mass;
        }
    }

    for(int row = 0; row < edgeCount; ++row) {
        expectTag(input, "EDGE");
        int edge = -1;
        int initialCut = 0;
        int entryCount = 0;
        input >> edge;
        if(edge < 0 || edge >= edgeCount) {
            throw std::runtime_error("Invalid EDGE id in footprint sidecar.");
        }
        Edge& value = result.edges_[edge];
        input >> value.vertex0 >> value.vertex1 >> value.face0 >> value.face1 >> initialCut >> entryCount;
        if(
            value.vertex0 < 0 || value.vertex1 < 0 || value.face0 < 0 || value.face1 < 0 ||
            value.face0 >= result.faceCount_ || value.face1 >= result.faceCount_ ||
            (initialCut != 0 && initialCut != 1) || entryCount < 0
        ) {
            throw std::runtime_error("Invalid EDGE record in footprint sidecar.");
        }
        result.cut_[edge] = static_cast<std::uint8_t>(initialCut);
        result.vertexPairToEdge_[edgeKey(value.vertex0, value.vertex1)] = edge;
        for(int entry = 0; entry < entryCount; ++entry) {
            int residue = -1;
            input >> residue;
            if(residue < 0 || residue >= residueCount) {
                throw std::runtime_error("Invalid EDGE residue id in footprint sidecar.");
            }
            value.residues.push_back(residue);
            result.residues_[residue].edges.push_back(edge);
        }
    }
    if(!input) {
        throw std::runtime_error("Truncated footprint sidecar.");
    }
    result.prepare();
    return result;
}

void ResidueFootprintEnergy::prepare() {
    weightSum_ = 0.0;
    for(int residueId = 0; residueId < static_cast<int>(residues_.size()); ++residueId) {
        Residue& residue = residues_[residueId];
        UnionFind baseline(static_cast<int>(residue.faces.size()));
        for(const int edgeId : residue.edges) {
            const Edge& edge = edges_[edgeId];
            baseline.unite(residue.faceToLocal.at(edge.face0), residue.faceToLocal.at(edge.face1));
        }
        residue.baselineComponent.resize(residue.faces.size());
        std::set<int> components;
        for(int localFace = 0; localFace < static_cast<int>(residue.faces.size()); ++localFace) {
            residue.baselineComponent[localFace] = baseline.find(localFace);
            components.insert(residue.baselineComponent[localFace]);
        }
        residue.cycleRank = std::max(
            0,
            static_cast<int>(residue.edges.size()) - static_cast<int>(residue.faces.size()) +
                static_cast<int>(components.size())
        );
        if(residue.totalMass > 0.0 && residue.objectiveWeight > 0.0) {
            weightSum_ += residue.objectiveWeight;
        }
    }

    residueScores_.resize(residues_.size(), 0.0);
    weightedScore_ = 0.0;
    for(int residue = 0; residue < static_cast<int>(residues_.size()); ++residue) {
        residueScores_[residue] = residueScore(residue);
        weightedScore_ += residues_[residue].objectiveWeight * residueScores_[residue];
    }
}

double ResidueFootprintEnergy::residueScore(
    int residueId,
    const std::unordered_map<int, bool>* overrides
) const {
    const Residue& residue = residues_[residueId];
    if(residue.totalMass <= 0.0 || residue.faces.empty()) {
        return 0.0;
    }

    UnionFind connected(static_cast<int>(residue.faces.size()));
    for(const int edgeId : residue.edges) {
        bool isCut = cut_[edgeId] != 0;
        if(overrides != nullptr) {
            const auto overrideValue = overrides->find(edgeId);
            if(overrideValue != overrides->end()) {
                isCut = overrideValue->second;
            }
        }
        if(!isCut) {
            const Edge& edge = edges_[edgeId];
            connected.unite(residue.faceToLocal.at(edge.face0), residue.faceToLocal.at(edge.face1));
        }
    }

    std::unordered_map<std::uint64_t, double> pieceMass;
    std::unordered_map<int, double> baselineMass;
    for(int localFace = 0; localFace < static_cast<int>(residue.faces.size()); ++localFace) {
        const int baseline = residue.baselineComponent[localFace];
        const int piece = connected.find(localFace);
        const double mass = residue.masses[localFace];
        const std::uint64_t key =
            (static_cast<std::uint64_t>(static_cast<std::uint32_t>(baseline)) << 32U) |
            static_cast<std::uint32_t>(piece);
        pieceMass[key] += mass;
        baselineMass[baseline] += mass;
    }

    std::unordered_map<int, double> squaredPieceMass;
    for(const auto& item : pieceMass) {
        const int baseline = static_cast<int>(item.first >> 32U);
        squaredPieceMass[baseline] += item.second * item.second;
    }
    double fragmentationMass = 0.0;
    for(const auto& item : baselineMass) {
        if(item.second > 0.0) {
            fragmentationMass += item.second - squaredPieceMass[item.first] / item.second;
        }
    }
    return fragmentationMass / residue.totalMass;
}

std::unordered_map<int, bool> ResidueFootprintEnergy::normalizedChanges(
    const std::vector<FootprintEdgeChange>& changes
) const {
    std::unordered_map<int, bool> result;
    for(const FootprintEdgeChange& change : changes) {
        if(change.edge < 0 || change.edge >= static_cast<int>(edges_.size())) {
            throw std::out_of_range("Footprint edge change is outside the edge table.");
        }
        result[change.edge] = change.cut;
    }
    return result;
}

std::vector<int> ResidueFootprintEnergy::touchedResidues(
    const std::unordered_map<int, bool>& changes
) const {
    std::unordered_set<int> touched;
    for(const auto& change : changes) {
        if((cut_[change.first] != 0) == change.second) {
            continue;
        }
        touched.insert(edges_[change.first].residues.begin(), edges_[change.first].residues.end());
    }
    std::vector<int> result(touched.begin(), touched.end());
    std::sort(result.begin(), result.end());
    return result;
}

double ResidueFootprintEnergy::candidateDelta(
    const std::vector<FootprintEdgeChange>& changes
) const {
    if(weightSum_ <= 0.0) {
        return 0.0;
    }
    const std::unordered_map<int, bool> normalized = normalizedChanges(changes);
    double delta = 0.0;
    for(const int residue : touchedResidues(normalized)) {
        delta += residues_[residue].objectiveWeight *
            (residueScore(residue, &normalized) - residueScores_[residue]);
    }
    return delta / weightSum_;
}

void ResidueFootprintEnergy::commit(const std::vector<FootprintEdgeChange>& changes) {
    const std::unordered_map<int, bool> normalized = normalizedChanges(changes);
    const std::vector<int> touched = touchedResidues(normalized);
    for(const auto& change : normalized) {
        cut_[change.first] = static_cast<std::uint8_t>(change.second);
    }
    for(const int residue : touched) {
        weightedScore_ -= residues_[residue].objectiveWeight * residueScores_[residue];
        residueScores_[residue] = residueScore(residue);
        weightedScore_ += residues_[residue].objectiveWeight * residueScores_[residue];
    }
}

void ResidueFootprintEnergy::synchronize(
    const std::vector<int>& faceVertices,
    const std::vector<int>& sourceVertices
) {
    if(faceVertices.size() != static_cast<std::size_t>(faceCount_) * 3U) {
        throw std::runtime_error("Current face table does not match the footprint sidecar.");
    }
    std::vector<FootprintEdgeChange> changes;
    changes.reserve(edges_.size());
    for(int edgeId = 0; edgeId < static_cast<int>(edges_.size()); ++edgeId) {
        const Edge& edge = edges_[edgeId];
        int current[2][2] = {{-1, -1}, {-1, -1}};
        const int faces[2] = {edge.face0, edge.face1};
        const int sources[2] = {edge.vertex0, edge.vertex1};
        for(int side = 0; side < 2; ++side) {
            for(int corner = 0; corner < 3; ++corner) {
                const int vertex = faceVertices[faces[side] * 3 + corner];
                if(vertex < 0 || vertex >= static_cast<int>(sourceVertices.size())) {
                    throw std::runtime_error("Current face contains an invalid vertex index.");
                }
                for(int endpoint = 0; endpoint < 2; ++endpoint) {
                    if(sourceVertices[vertex] == sources[endpoint]) {
                        current[side][endpoint] = vertex;
                    }
                }
            }
        }
        if(
            current[0][0] < 0 || current[0][1] < 0 ||
            current[1][0] < 0 || current[1][1] < 0
        ) {
            throw std::runtime_error("Current faces no longer contain an original footprint edge.");
        }
        const bool cut = current[0][0] != current[1][0] || current[0][1] != current[1][1];
        if((cut_[edgeId] != 0) != cut) {
            changes.push_back(FootprintEdgeChange(edgeId, cut));
        }
    }
    commit(changes);
}

double ResidueFootprintEnergy::score() const {
    return weightSum_ > 0.0 ? weightedScore_ / weightSum_ : 0.0;
}

int ResidueFootprintEnergy::edgeId(int vertex0, int vertex1) const {
    const auto found = vertexPairToEdge_.find(edgeKey(vertex0, vertex1));
    return found == vertexPairToEdge_.end() ? -1 : found->second;
}

int ResidueFootprintEnergy::cycleRank() const {
    int result = 0;
    for(const Residue& residue : residues_) {
        result += residue.cycleRank;
    }
    return result;
}

int ResidueFootprintEnergy::residueCount() const {
    return static_cast<int>(residues_.size());
}

int ResidueFootprintEnergy::edgeCount() const {
    return static_cast<int>(edges_.size());
}

const std::vector<int>& ResidueFootprintEnergy::inputSourceVertices() const {
    return inputSourceVertices_;
}

} // namespace TopoPPI
