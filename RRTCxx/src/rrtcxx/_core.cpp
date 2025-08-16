#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include "coal/math/transform.h"
#include "coal/mesh_loader/loader.h"
#include "coal/BVH/BVH_model.h"
#include "coal/collision.h"
#include "coal/collision_data.h"
#include "coal/distance.h"

#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include <random>
#include <memory>
#include <chrono>
#include <limits>
#include <cmath>
#include <algorithm>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

// -------------------- State --------------------
struct State {
    Eigen::Vector3d position;
    Eigen::Quaterniond orientation;

    State()
        : position(Eigen::Vector3d::Zero()), orientation(Eigen::Quaterniond::Identity()) {}

    State(const Eigen::Vector3d& pos, const Eigen::Quaterniond& quat)
        : position(pos), orientation(quat) {}

    // A: (3,1) pos, B: (4,1) quat (x,y,z,w) but Eigen ctor uses (w,x,y,z)
    State(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B) {
        position = Eigen::Vector3d(A(0,0), A(1,0), A(2,0));
        orientation = Eigen::Quaterniond(B(3,0), B(0,0), B(1,0), B(2,0));
    }

    bool operator==(const State& other) const {
        return position.isApprox(other.position) && orientation.isApprox(other.orientation);
    }

    double distance(const State& other) const {
        const double pos_dist = (position - other.position).norm();
        const double qdot = orientation.dot(other.orientation);
        const double quat_dist = 1.0 - qdot*qdot; // in [0,1]
        return pos_dist + quat_dist;
    }

    State interpolate(const State& other, double alpha) const {
        Eigen::Vector3d p = (1.0 - alpha) * position + alpha * other.position;
        Eigen::Quaterniond q = orientation.slerp(alpha, other.orientation);
        return State(p, q);
    }
};

// -------------------- Node --------------------
struct Node {
    State state;
    Node* parent;
    double cost_from_root; // for RRT*

    Node(const State& s, Node* p = nullptr, double cost = 0.0)
        : state(s), parent(p), cost_from_root(cost) {}
};

// -------------------- RRTPlanner3D --------------------
class RRTPlanner3D {
public:
    RRTPlanner3D(Eigen::MatrixXd vertexLocation,
                 Eigen::MatrixXi triangleIndices,
                 Eigen::Vector3d payloadTranslation = Eigen::Vector3d::Zero(),
                 Eigen::Vector3d payloadSize = Eigen::Vector3d::Zero(),
                 bool usePayload = false,
                 int numIterations = 10000,
                 double stepSize = 0.5,
                 double goalBias = 0.1,
                 double minX = -10.0, double maxX = 10.0,
                 double minY = -10.0, double maxY = 10.0,
                 double minZ = -10.0, double maxZ = 10.0)
        : vertexLocation(vertexLocation),
          triangleIndices(triangleIndices),
          payloadTranslation(payloadTranslation),
          payloadSize(payloadSize),
          usePayload(usePayload),
          numIterations(numIterations),
          stepSize(stepSize),
          goalBias(goalBias),
          minX(minX), maxX(maxX),
          minY(minY), maxY(maxY),
          minZ(minZ), maxZ(maxZ)
    {
        setupEnvironment();
        this->robotBox = createBox(0.45, 0.45, 0.12);
        this->payloadBox = createBox(payloadSize);

        // RRT* defaults
        useKNearest_ = false;
        kNearest_ = 15;
        useRStarRadius_ = true;
        rewireRadius_ = 1.5;
        gammaRStar_ = 2.0;
        dimension_ = 6; // SE(3)
    }

    // ---------- Accessors for trees (debug) ----------
    std::vector<Node*> getTreeA() const { return treeA; }
    std::vector<Node*> getTreeB() const { return treeB; }

    // ---------- Tunable setters (exposed to Python) ----------
    void setUseKNearest(bool v) { useKNearest_ = v; }
    void setKNearest(int k) { kNearest_ = std::max(1, k); }
    void setUseRStarRadius(bool v) { useRStarRadius_ = v; }
    void setRewireRadius(double r) { rewireRadius_ = std::max(0.0, r); }
    void setGammaRStar(double g) { gammaRStar_ = std::max(0.0, g); }
    void setDimension(int d) { dimension_ = std::max(1, d); }

    // ---------- Existing Bi-RRT (returns (path, seconds)) ----------
    std::pair<std::vector<State>, double> plan_birrt(const State& start, const State& goal) {
        auto t0 = std::chrono::steady_clock::now();

        std::vector<Node*> ta; // start tree
        std::vector<Node*> tb; // goal tree

        this->Distance(start);
        this->Distance(goal);

        ta.push_back(new Node(start, nullptr, 0.0));
        tb.push_back(new Node(goal, nullptr, 0.0));

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> distX(minX, maxX);
        std::uniform_real_distribution<> distY(minY, maxY);
        std::uniform_real_distribution<> distZ(minZ, maxZ);
        std::uniform_real_distribution<> distBias(0.0, 1.0);

        auto connect_toward = [&](std::vector<Node*>& tree, const State& target) -> Node* {
            Node* nearest = findNearest(tree, target);
            State stepState = steer(nearest->state, target);
            if (!isMotionValid(nearest->state, stepState)) return nullptr;
            Node* last = new Node(stepState, nearest, nearest->cost_from_root + edgeCost(nearest->state, stepState));
            tree.push_back(last);

            int guard = 1000;
            while (last->state.distance(target) >= stepSize && guard-- > 0) {
                State next = steer(last->state, target);
                if (!isMotionValid(last->state, next)) return last;
                Node* nn = new Node(next, last, last->cost_from_root + edgeCost(last->state, next));
                tree.push_back(nn);
                last = nn;
            }
            return last;
        };

        for (int i = 0; i < numIterations; ++i) {
            State q_rand = (distBias(gen) < goalBias)
                ? State(goal.position, goal.orientation)
                : State(Eigen::Vector3d(distX(gen), distY(gen), distZ(gen)), Eigen::Quaterniond::UnitRandom());

            Node* qa_last = connect_toward(ta, q_rand);
            if (!qa_last) { std::swap(ta, tb); continue; }

            Node* qb_last = connect_toward(tb, qa_last->state);

            if (qb_last && qb_last->state.distance(qa_last->state) < stepSize
                && isMotionValid(qb_last->state, qa_last->state)) {

                Node* bridgeB = new Node(qa_last->state, qb_last, qb_last->cost_from_root + edgeCost(qb_last->state, qa_last->state));
                tb.push_back(bridgeB);

                std::vector<State> path = constructPath(qa_last, bridgeB);
                this->treeA = ta; this->treeB = tb;

                auto t1 = std::chrono::steady_clock::now();
                double seconds = std::chrono::duration<double>(t1 - t0).count();
                return {path, seconds};
            }
            std::swap(ta, tb);
        }

        this->treeA = ta; this->treeB = tb;
        auto t1 = std::chrono::steady_clock::now();
        double seconds = std::chrono::duration<double>(t1 - t0).count();
        return {{}, seconds};
    }

    // ---------- Existing single-tree Bi-RRT variant (kept) ----------
    std::pair<std::vector<State>, double> plan(const State& start, const State& goal) {
        auto t0 = std::chrono::steady_clock::now();

        std::vector<Node*> treeA;
        std::vector<Node*> treeB;

        this->Distance(start);
        this->Distance(goal);

        treeA.push_back(new Node(start, nullptr, 0.0));
        treeB.push_back(new Node(goal, nullptr, 0.0));

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> distX(minX, maxX);
        std::uniform_real_distribution<> distY(minY, maxY);
        std::uniform_real_distribution<> distZ(minZ, maxZ);
        std::uniform_real_distribution<> distBias(0.0, 1.0);

        for (int i = 0; i < numIterations; ++i) {
            State randState = (distBias(gen) < goalBias)
                ? goal
                : State(Eigen::Vector3d(distX(gen), distY(gen), distZ(gen)), Eigen::Quaterniond::UnitRandom());

            Node* nearestA = findNearest(treeA, randState);
            State newStateA = steer(nearestA->state, randState);
            if (!isMotionValid(nearestA->state, newStateA)) { std::swap(treeA, treeB); continue; }

            Node* newNodeA = new Node(newStateA, nearestA, nearestA->cost_from_root + edgeCost(nearestA->state, newStateA));
            treeA.push_back(newNodeA);

            Node* nearestB = findNearest(treeB, newNodeA->state);
            State newStateB = steer(nearestB->state, newNodeA->state);

            if (isMotionValid(nearestB->state, newStateB)) {
                Node* newNodeB = new Node(newStateB, nearestB, nearestB->cost_from_root + edgeCost(nearestB->state, newStateB));
                treeB.push_back(newNodeB);

                if (newStateB.distance(newNodeA->state) < stepSize) {
                    std::vector<State> path = constructPath(newNodeA, newNodeB);
                    auto t1 = std::chrono::steady_clock::now();
                    double seconds = std::chrono::duration<double>(t1 - t0).count();
                    return {path, seconds};
                }
            }
            std::swap(treeA, treeB);
        }

        this->treeA = treeA; this->treeB = treeB;
        auto t1 = std::chrono::steady_clock::now();
        double seconds = std::chrono::duration<double>(t1 - t0).count();
        return {{}, seconds};
    }

    // ---------- NEW: Bi-RRT* (returns (path, seconds)) ----------
    std::pair<std::vector<State>, double> plan_birrt_star(const State& start, const State& goal) {
        auto t0 = std::chrono::steady_clock::now();

        std::vector<Node*> ta; // start tree
        std::vector<Node*> tb; // goal tree
        ta.push_back(new Node(start, nullptr, 0.0));
        tb.push_back(new Node(goal, nullptr, 0.0));

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> distX(minX, maxX);
        std::uniform_real_distribution<> distY(minY, maxY);
        std::uniform_real_distribution<> distZ(minZ, maxZ);
        std::uniform_real_distribution<> distBias(0.0, 1.0);

        auto extend_star = [&](std::vector<Node*>& tree, const State& q_target) -> Node* {
            Node* q_near = findNearest(tree, q_target);
            State q_new_state = steer(q_near->state, q_target);
            if (!isMotionValid(q_near->state, q_new_state)) return nullptr;

            // choose best parent among neighbors
            auto neigh = getNeighbors(tree, q_new_state);
            Node* best_parent = q_near;
            double best_cost = q_near->cost_from_root + edgeCost(q_near->state, q_new_state);

            for (Node* n : neigh) {
                double c = n->cost_from_root + edgeCost(n->state, q_new_state);
                if (c + 1e-12 < best_cost && isMotionValid(n->state, q_new_state)) {
                    best_cost = c;
                    best_parent = n;
                }
            }

            Node* q_new = new Node(q_new_state, best_parent, best_cost);
            tree.push_back(q_new);

            // rewire neighbors through q_new if cost improves
            for (Node* n : neigh) {
                double c_through_new = q_new->cost_from_root + edgeCost(q_new->state, n->state);
                if (c_through_new + 1e-12 < n->cost_from_root && isMotionValid(q_new->state, n->state)) {
                    n->parent = q_new;
                    n->cost_from_root = c_through_new;
                }
            }
            return q_new;
        };

        for (int i = 0; i < numIterations; ++i) {
            State q_rand = (distBias(gen) < goalBias)
                ? State(goal.position, goal.orientation)
                : State(Eigen::Vector3d(distX(gen), distY(gen), distZ(gen)), Eigen::Quaterniond::UnitRandom());

            Node* qa = extend_star(ta, q_rand);
            if (!qa) { std::swap(ta, tb); continue; }

            Node* qb = extend_star(tb, qa->state);
            if (qb && qb->state.distance(qa->state) < stepSize
                && isMotionValid(qb->state, qa->state)) {

                Node* bridge = new Node(qa->state, qb, qb->cost_from_root + edgeCost(qb->state, qa->state));
                tb.push_back(bridge);
                std::vector<State> path = constructPath(qa, bridge);

                auto t1 = std::chrono::steady_clock::now();
                double seconds = std::chrono::duration<double>(t1 - t0).count();
                return {path, seconds};
            }
            std::swap(ta, tb);
        }

        auto t1 = std::chrono::steady_clock::now();
        double seconds = std::chrono::duration<double>(t1 - t0).count();
        return {{}, seconds};
    }

    // ---------- Path pruning (unchanged) ----------
    std::vector<State> prunePath (const std::vector<State>& path) {
        if (path.size() < 3) return path;
        std::vector<State> pruned; pruned.push_back(path.front());

        size_t i = 0, n = path.size();
        while (i < n - 1) {
            size_t j = n - 1;
            while (j > i + 1) {
                if (isMotionValid(path[i], path[j])) break;
                --j;
            }
            pruned.push_back(path[j]);
            i = j;
        }

        std::vector<State> interp;
        for (size_t k = 0; k + 1 < pruned.size(); ++k) {
            const State& a = pruned[k];
            const State& b = pruned[k+1];
            double d = a.distance(b);
            int steps = std::max(1, int(d / stepSize));
            for (int s = 0; s < steps; ++s) {
                double alpha = double(s) / steps;
                interp.push_back(a.interpolate(b, alpha));
            }
        }
        interp.push_back(pruned.back());
        return interp;
    }

private:
    // environment
    std::shared_ptr<coal::BVHModel<coal::OBBRSS>> envModel;
    std::shared_ptr<coal::Box> robotBox;
    std::shared_ptr<coal::Box> payloadBox;
    Eigen::MatrixXd vertexLocation;
    Eigen::MatrixXi triangleIndices;

    // trees (debug)
    std::vector<Node*> treeA, treeB;

    // params
    bool usePayload;
    Eigen::Vector3d payloadTranslation;
    Eigen::Vector3d payloadSize;

    int numIterations;
    double stepSize, goalBias;
    double minX, maxX, minY, maxY, minZ, maxZ;

    // RRT* tunables
    bool useKNearest_;
    int  kNearest_;
    bool useRStarRadius_;
    double rewireRadius_;
    double gammaRStar_;
    int dimension_;

    // ---- utils ----
    void setupEnvironment() {
        envModel = std::make_shared<coal::BVHModel<coal::OBBRSS>>();
        envModel->beginModel(vertexLocation.rows(), triangleIndices.rows());
        Eigen::Matrix<double, Eigen::Dynamic, 3> V = vertexLocation;
        Eigen::Matrix<long,   Eigen::Dynamic, 3> F = triangleIndices.cast<long>();
        envModel->addVertices(V);
        envModel->addTriangles(F);
        envModel->endModel();
    }

    std::shared_ptr<coal::Box> createBox(double x, double y, double z) {
        return std::make_shared<coal::Box>(x, y, z);
    }
    std::shared_ptr<coal::Box> createBox(const Eigen::Vector3d& s) {
        return std::make_shared<coal::Box>(s.x(), s.y(), s.z());
    }

    Node* findNearest(const std::vector<Node*>& tree, const State& state) {
        Node* nearest = nullptr;
        double best = std::numeric_limits<double>::max();
        for (auto* node : tree) {
            double d = node->state.distance(state);
            if (d < best) { best = d; nearest = node; }
        }
        return nearest;
    }

    State steer(const State& from, const State& to) {
        double d = from.distance(to);
        double a = std::min(stepSize / std::max(d, 1e-12), 1.0);
        return from.interpolate(to, a);
    }

    double edgeCost(const State& a, const State& b) const {
        return a.distance(b); // customize if you want orientation/clearance weights
    }

    bool isCollisionFree(const State& s) {
        coal::Transform3s T1, T2;
        T1.setQuatRotation(s.orientation);
        T1.setTranslation(s.position);
        T2.setQuatRotation(Eigen::Quaterniond::Identity());
        T2.setTranslation(Eigen::Vector3d::Zero());

        coal::CollisionRequest req;
        coal::CollisionResult res;
        coal::collide(robotBox.get(), T1, envModel.get(), T2, req, res);
        bool ok = !res.isCollision();
        res.clear();

        if (ok && usePayload) {
            coal::Transform3s Tp;
            Tp.setQuatRotation(s.orientation);
            Tp.setTranslation(s.position + payloadTranslation);
            coal::collide(payloadBox.get(), Tp, envModel.get(), T2, req, res);
            ok = ok && !res.isCollision();
            res.clear();
        }
        return ok;
    }

    void Distance(const State& s) {
        coal::Transform3s T1, T2;
        T1.setQuatRotation(s.orientation);
        T1.setTranslation(s.position);
        T2.setQuatRotation(Eigen::Quaterniond::Identity());
        T2.setTranslation(Eigen::Vector3d::Zero());
        coal::DistanceRequest req;
        coal::DistanceResult res;
        coal::distance(robotBox.get(), T1, envModel.get(), T2, req, res);
    }

    bool isMotionValid(const State& a, const State& b) {
        double d = a.distance(b);
        int steps = std::max(1, int(d / stepSize));
        for (int i = 0; i <= steps; ++i) {
            double alpha = double(i) / steps;
            State s = a.interpolate(b, alpha);
            if (!isCollisionFree(s)) return false;
        }
        return true;
    }

    // Neighbor selection for RRT*
    std::vector<Node*> getNeighbors(const std::vector<Node*>& tree, const State& q_new) {
        std::vector<Node*> neigh;
        size_t n = tree.size();
        if (n == 0) return neigh;

        if (useKNearest_) {
            // k-nearest
            const int k = std::min<int>(kNearest_, int(n));
            std::vector<std::pair<double, Node*>> arr; arr.reserve(n);
            for (Node* node : tree) {
                arr.emplace_back(node->state.distance(q_new), node);
            }
            std::nth_element(arr.begin(), arr.begin() + (k-1), arr.end(),
                             [](auto& a, auto& b){ return a.first < b.first; });
            neigh.reserve(k);
            for (int i = 0; i < k; ++i) neigh.push_back(arr[i].second);
        } else {
            // radius-based
            double radius;
            if (useRStarRadius_) {
                double nn = std::max<size_t>(1, n);
                radius = gammaRStar_ * std::pow(std::log((double)nn) / (double)nn, 1.0 / std::max(1, dimension_));
                radius = std::max(radius, stepSize); // practical floor
            } else {
                radius = rewireRadius_;
            }
            for (Node* node : tree) {
                if (node->state.distance(q_new) <= radius) {
                    neigh.push_back(node);
                }
            }
            // ensure at least the nearest is included
            if (neigh.empty()) {
                Node* nearest = findNearest(tree, q_new);
                if (nearest) neigh.push_back(nearest);
            }
        }
        return neigh;
    }

    std::vector<State> constructPath(Node* a_tip, Node* b_tip) {
        std::vector<State> path1, path2;
        for (Node* n = a_tip; n != nullptr; n = n->parent) path1.push_back(n->state);
        for (Node* n = b_tip; n != nullptr; n = n->parent) path2.push_back(n->state);
        std::reverse(path1.begin(), path1.end());
        path1.insert(path1.end(), path2.begin(), path2.end());
        return path1;
    }
};

// Python bindings
namespace py = pybind11;

PYBIND11_MODULE(_core, m) {
    m.doc() = R"pbdoc(
        rrtcxx core with BiRRT and BiRRT*
    )pbdoc";

    py::class_<State>(m, "State")
        .def(py::init<>())
        .def(py::init<const Eigen::Vector3d&, const Eigen::Quaterniond&>())
        .def(py::init<const Eigen::MatrixXd&, const Eigen::MatrixXd&>())
        .def_readwrite("position", &State::position)
        .def_readwrite("orientation", &State::orientation)
        .def("distance", &State::distance)
        .def("interpolate", &State::interpolate)
        .def_property_readonly("q", [](const State& self) {
            return Eigen::Vector4d(
                self.orientation.x(),
                self.orientation.y(),
                self.orientation.z(),
                self.orientation.w()
            );
        });

    py::class_<Node>(m, "Node")
        .def_readwrite("state", &Node::state)
        .def_readwrite("parent", &Node::parent)
        .def_readwrite("cost_from_root", &Node::cost_from_root);

    py::class_<RRTPlanner3D>(m, "RRTPlanner3D")
        .def(py::init<Eigen::MatrixXd, Eigen::MatrixXi, Eigen::Vector3d, Eigen::Vector3d, bool, int, double, double, double, double, double, double, double, double>())

        // Tunable setters
        .def("setUseKNearest", &RRTPlanner3D::setUseKNearest)
        .def("setKNearest", &RRTPlanner3D::setKNearest)
        .def("setUseRStarRadius", &RRTPlanner3D::setUseRStarRadius)
        .def("setRewireRadius", &RRTPlanner3D::setRewireRadius)
        .def("setGammaRStar", &RRTPlanner3D::setGammaRStar)
        .def("setDimension", &RRTPlanner3D::setDimension)

        // Planners
        .def("plan", &RRTPlanner3D::plan)                     // (path, seconds)
        .def("biRRT", &RRTPlanner3D::plan_birrt)              // (path, seconds)
        .def("biRRTStar", &RRTPlanner3D::plan_birrt_star)     // (path, seconds)

        // Utilities
        .def("getTreeA", &RRTPlanner3D::getTreeA)
        .def("getTreeB", &RRTPlanner3D::getTreeB)
        .def("prunePath", &RRTPlanner3D::prunePath);

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
