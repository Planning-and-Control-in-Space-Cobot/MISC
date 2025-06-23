#include <pybind11/pybind11.h>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include <random>

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
            std::cout << "State created with position: " << pos.transpose()
                      << " and orientation: " << quat.coeffs().transpose() << std::endl;
        }

    // Always assume A (3,) for pos, B (4,) for quat
    State(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B) {
        // Just map first 3 for pos, first 4 for quat
        position = Eigen::Vector3d(A(0, 0), A(1, 0), A(2, 0));
        orientation = Eigen::Quaterniond(B(3, 0), B(0, 0), B(1, 0), B(2, 0));  // Eigen uses (w,x,y,z)

        std::cout << "State created from matrices with position: " << position.transpose()
                  << " and orientation: " << orientation.coeffs().transpose() << std::endl;
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

    Node(const State& state_, Node* parent_ = nullptr)
        : state(state_), parent(parent_) {}
};

// -------------------- RRT --------------------
std::vector<State> plan_rrt(const State& start, const State& goal, int num_iterations) {
    std::vector<Node*> nodes;
    nodes.push_back(new Node(start));

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dist(-10.0, 10.0);

    for (int i = 0; i < num_iterations; ++i) {
        // Sample random A (3x1) and B (4x1)
        Eigen::MatrixXd A(3, 1);
        Eigen::MatrixXd B(4, 1);
        A << dist(gen), dist(gen), dist(gen);
        B << dist(gen), dist(gen), dist(gen), 1.0;  // make a unit quaternion roughly
        State rand_state(A, B);

        // Find nearest
        Node* nearest = nodes[0];
        double min_dist = nearest->state.distance(rand_state);
        for (auto* n : nodes) {
            double d = n->state.distance(rand_state);
            if (d < min_dist) {
                min_dist = d;
                nearest = n;
            }
        }

        // Interpolate towards sample
        State new_state = nearest->state.interpolate(rand_state, 0.1);
        Node* new_node = new Node(new_state, nearest);
        nodes.push_back(new_node);

        // Goal check
        if (new_state.distance(goal) < 0.5) {
            std::vector<State> path;
            for (Node* n = new_node; n != nullptr; n = n->parent) {
                path.push_back(n->state);
            }
            std::reverse(path.begin(), path.end());

            for (auto* n : nodes) delete n;
            return path;
        }
    }

    for (auto* n : nodes) delete n;
    return {};
}

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
        .def("interpolate", &State::interpolate);

    py::class_<Node>(m, "Node")
        .def_readwrite("state", &Node::state)
        .def_readwrite("parent", &Node::parent);

    
    m.def("plan_rrt", &plan_rrt, "Plan RRT path");

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}