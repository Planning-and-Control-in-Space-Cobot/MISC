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

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

// -------------------- State --------------------
struct State {
    Eigen::Vector3d position;
    Eigen::Quaterniond orientation;

    State()
        : position(Eigen::Vector3d::Zero()), orientation(Eigen::Quaterniond::Identity()) {}

    // Native C++ constructor
    State(const Eigen::Vector3d& pos, const Eigen::Quaterniond& quat)
        : position(pos), orientation(quat) {
        }

    // Always assume A (3,) for pos, B (4,) for quat
    State(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B) {
        // Just map first 3 for pos, first 4 for quat
        position = Eigen::Vector3d(A(0, 0), A(1, 0), A(2, 0));
        orientation = Eigen::Quaterniond(B(3, 0), B(0, 0), B(1, 0), B(2, 0));  // Eigen uses (w,x,y,z)
    }

    bool operator==(const State& other) const {
        return position.isApprox(other.position) && orientation.isApprox(other.orientation);
    }

    double distance(const State& other) const {
        double pos_dist = (position - other.position).norm();
        double quat_dist = 1.0 - orientation.dot(other.orientation);
        return pos_dist + quat_dist;
    }

    State interpolate(const State& other, double alpha) const {
        Eigen::Vector3d interp_pos = (1 - alpha) * position + alpha * other.position;
        Eigen::Quaterniond interp_quat = orientation.slerp(alpha, other.orientation);
        return State(interp_pos, interp_quat);
    }
};

// -------------------- Node --------------------
struct Node {
    State state;
    Node* parent;
    Node(const State& s, Node* p = nullptr) : state(s), parent(p) {}
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
        this->robotBox = createBox(0.45, 0.45, 0.12); // simple robot box
        this->payloadBox = createBox(payloadSize); // payload box
    }

    std::vector<Node *> getTreeA() const {
        return treeA;
    }
    
    std::vector<Node *> getTreeB() const {
        return treeB;
    }

    std::vector<State> plan(const State& start, const State& goal) {
        std::vector<Node*> treeA;
        std::vector<Node*> treeB;

        this->Distance(start);
        this->Distance(goal);

        treeA.push_back(new Node(start));
        treeB.push_back(new Node(goal));

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> distX(minX, maxX);
        std::uniform_real_distribution<> distY(minY, maxY);
        std::uniform_real_distribution<> distZ(minZ, maxZ);
        std::uniform_real_distribution<> distBias(0.0, 1.0);

        for (int i = 0; i < numIterations; ++i) {
            // 1) Sample random
            State randState;
            if (distBias(gen) < goalBias) {
                randState = goal;
            } else {
                randState = State(
                    Eigen::Vector3d(distX(gen), distY(gen), distZ(gen)),
                    Eigen::Quaterniond::UnitRandom());
            }

            // 2) Extend treeA towards random sample
            Node* nearestA = findNearest(treeA, randState);
            State newStateA = steer(nearestA->state, randState);

            if (!isMotionValid(nearestA->state, newStateA)) {
                //std::cout << "Motion invalid from " << nearestA->state.position.transpose() 
                  //        << " to " << newStateA.position.transpose() << std::endl;
                continue;
            }
            //std::cout << "Motion valid from " << nearestA->state.position.transpose() 
             //         << " to " << newStateA.position.transpose() << " start tree size " << treeA.size() << " goal tree size " << treeB.size() << std::endl;

            Node* newNodeA = new Node(newStateA, nearestA);
            treeA.push_back(newNodeA);

            // 3) Try to connect treeB to new node
            Node* nearestB = findNearest(treeB, newNodeA->state);
            State newStateB = steer(nearestB->state, newNodeA->state);

            if (isMotionValid(nearestB->state, newStateB)) {
                Node* newNodeB = new Node(newStateB, nearestB);
                treeB.push_back(newNodeB);

                // Check if they connected directly
                if (newStateB.distance(newNodeA->state) < stepSize) {
                    // Success: build path
                    std::vector<State> path = constructPath(newNodeA, newNodeB);
                    std::cerr << "Path found in " << i << " iterations." << std::endl;

                    for (auto const & node : path)  {
                        auto pos = node.position;
                        auto quat = node.orientation;
                        auto collision = isCollisionFree(node);

                        std::cout << "Position: " << pos.transpose()
                                    << ", Orientation: " << quat.coeffs().transpose()
                                    << ", Collision Free: " << collision << std::endl;

                    }


                    return path;
                }
            }

            // 4) Swap trees
            std::swap(treeA, treeB);
        }

        std::cerr << "Failed to find path!" << std::endl;
        this->treeA = treeA;
        this->treeB = treeB;
        return {};
    }

private:
    std::shared_ptr<coal::BVHModel<coal::OBBRSS>> envModel;
    std::shared_ptr<coal::Box> robotBox;
    std::shared_ptr<coal::Box> payloadBox;
    Eigen::MatrixXd vertexLocation;
    Eigen::MatrixXi triangleIndices;
    std::vector<Node *> treeA, treeB;

    bool usePayload;
    Eigen::Vector3d payloadTranslation;
    Eigen::Vector3d payloadSize;

    int numIterations;
    double stepSize, goalBias;
    double minX, maxX, minY, maxY, minZ, maxZ;

    void setupEnvironment() {

        
        envModel = std::make_shared<coal::BVHModel<coal::OBBRSS>>();
        envModel->beginModel(vertexLocation.rows(), triangleIndices.rows());
        // EXPLICIT CAST for robust type:
        Eigen::Matrix<double, Eigen::Dynamic, 3> V = vertexLocation;
        Eigen::Matrix<long, Eigen::Dynamic, 3> F = triangleIndices.cast<long>();

        envModel->addVertices(V);
        envModel->addTriangles(F);
        envModel->endModel();
    }

    std::shared_ptr<coal::Box> createBox(double x, double y, double z) {
        return std::make_shared<coal::Box>(x, y, z);
    }

    std::shared_ptr<coal::Box> createBox(const Eigen::Vector3d& size) {
        return std::make_shared<coal::Box>(size.x(), size.y(), size.z());
    }

    Node* findNearest(const std::vector<Node*>& tree, const State& state) {
        Node* nearest = nullptr;
        double minDist = std::numeric_limits<double>::max();
        for (auto* node : tree) {
            double dist = node->state.distance(state);
            if (dist < minDist) {
                minDist = dist;
                nearest = node;
            }
        }
        return nearest;
    }

    State steer(const State& from, const State& to) {
        double dist = from.distance(to);
        double alpha = std::min(stepSize / dist, 1.0);
        return from.interpolate(to, alpha);
    }

    bool isCollisionFree(const State& state) {
        coal::Transform3s T1, T2;
        T1.setQuatRotation(state.orientation);
        T1.setTranslation(state.position);
        T2.setQuatRotation(Eigen::Quaterniond::Identity());
        T2.setTranslation(Eigen::Vector3d::Zero());
        coal::CollisionRequest req;
        coal::CollisionResult res;
        coal::collide(robotBox.get(), T1, envModel.get(), T2, req, res);
        
        bool collisionFree = !res.isCollision();
        res.clear();
        if (usePayload) {
            coal::Transform3s TPayload;
            TPayload.setQuatRotation(state.orientation);
            TPayload.setTranslation(state.position + payloadTranslation);
            coal::collide(payloadBox.get(), TPayload, envModel.get(), T2, req, res);
            collisionFree &= !res.isCollision();
            res.clear();
        }
        return collisionFree;
    }

    void Distance(const State& from) {
        coal::Transform3s T1, T2;
        T1.setQuatRotation(from.orientation);
        T1.setTranslation(from.position);
        T2.setQuatRotation(Eigen::Quaterniond::Identity());
        T2.setTranslation(Eigen::Vector3d::Zero());
        coal::DistanceRequest req;
        coal::DistanceResult res;

        coal::distance(robotBox.get(), T1, envModel.get(), T2, req, res);
    }
    
    bool isMotionValid(const State& from, const State& to) {
        double dist = from.distance(to);
        int steps = std::max(1, int(dist / stepSize));
        for (int i = 0; i <= steps; ++i) {
            double alpha = double(i) / steps;
            State interp = from.interpolate(to, alpha);
            if (!isCollisionFree(interp)) return false;
        }
        return true;
    }

    std::vector<State> constructPath(Node* startConnect, Node* goalConnect) {
        std::vector<State> path1, path2;
        for (Node* n = startConnect; n != nullptr; n = n->parent) path1.push_back(n->state);
        for (Node* n = goalConnect; n != nullptr; n = n->parent) path2.push_back(n->state);
        std::reverse(path1.begin(), path1.end());
        path1.insert(path1.end(), path2.begin(), path2.end());
        return path1;
    }
};

// Python bindings
namespace py = pybind11;

PYBIND11_MODULE(_core, m) {
    m.doc() = R"pbdoc(
        Pybind11 example plugin
        -----------------------

        .. currentmodule:: rrtcxx

        .. autosummary::
        :toctree: _generate

        add
        subtract
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
                self.orientation.w(),
                self.orientation.x(),
                self.orientation.y(),
                self.orientation.z()
            );
        });

    py::class_<Node>(m, "Node")
        .def_readwrite("state", &Node::state)
        .def_readwrite("parent", &Node::parent);


    py::class_<RRTPlanner3D>(m, "RRTPlanner3D")
        .def(py::init<Eigen::MatrixXd, Eigen::MatrixXi, Eigen::Vector3d, Eigen::Vector3d, bool, int, double, double, double, double, double, double, double, double>())
        .def("plan", &RRTPlanner3D::plan)
        .def("getTreeA", &RRTPlanner3D::getTreeA)
        .def("getTreeB", &RRTPlanner3D::getTreeB);

    

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}