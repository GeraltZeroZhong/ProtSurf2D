#ifndef TOPOPPI_RESIDUE_FOOTPRINT_ENERGY_HPP
#define TOPOPPI_RESIDUE_FOOTPRINT_ENERGY_HPP

#include <cstdint>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace TopoPPI {

struct FootprintEdgeChange {
    FootprintEdgeChange(int edgeValue, bool cutValue) : edge(edgeValue), cut(cutValue) {}

    int edge;
    bool cut;
};

class ResidueFootprintEnergy {
public:
    static ResidueFootprintEnergy load(const std::string& path);

    double score() const;
    double candidateDelta(const std::vector<FootprintEdgeChange>& changes) const;
    void commit(const std::vector<FootprintEdgeChange>& changes);
    void synchronize(
        const std::vector<int>& faceVertices,
        const std::vector<int>& sourceVertices
    );

    int edgeId(int vertex0, int vertex1) const;
    int cycleRank() const;
    int residueCount() const;
    int edgeCount() const;
    const std::vector<int>& inputSourceVertices() const;

private:
    struct Residue {
        double objectiveWeight = 0.0;
        double totalMass = 0.0;
        int cycleRank = 0;
        std::vector<int> faces;
        std::vector<double> masses;
        std::vector<int> edges;
        std::vector<int> baselineComponent;
        std::unordered_map<int, int> faceToLocal;
    };

    struct Edge {
        int vertex0 = -1;
        int vertex1 = -1;
        int face0 = -1;
        int face1 = -1;
        std::vector<int> residues;
    };

    double residueScore(
        int residue,
        const std::unordered_map<int, bool>* overrides = nullptr
    ) const;
    std::unordered_map<int, bool> normalizedChanges(
        const std::vector<FootprintEdgeChange>& changes
    ) const;
    std::vector<int> touchedResidues(
        const std::unordered_map<int, bool>& changes
    ) const;
    void prepare();

    int faceCount_ = 0;
    std::vector<int> inputSourceVertices_;
    std::vector<Residue> residues_;
    std::vector<Edge> edges_;
    std::vector<std::uint8_t> cut_;
    std::vector<double> residueScores_;
    std::unordered_map<std::uint64_t, int> vertexPairToEdge_;
    double weightSum_ = 0.0;
    double weightedScore_ = 0.0;
};

} // namespace TopoPPI

#endif
