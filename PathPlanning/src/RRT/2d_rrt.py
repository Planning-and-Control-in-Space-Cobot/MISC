import numpy as np
import casadi as ca
from scipy.spatial.transform import Rotation as R
from dataclasses import dataclass
import matplotlib.pyplot as plt

# === Estado SE(3) ===
@dataclass
class State:
    x: np.ndarray  # posição (3,)
    v: np.ndarray  # velocidade linear (3,)
    q: np.ndarray  # quaternion (4,)
    w: np.ndarray  # velocidade angular (3,)

    def as_array(self):
        return np.concatenate([self.x, self.v, self.q, self.w])

    def distance_to(self, other: 'State'):
        pos_dist = np.linalg.norm(self.x - other.x)
        quat_dist = 1 - abs(np.dot(self.q, other.q))
        return pos_dist + quat_dist

class Node:
    def __init__(self, state: State, parent=None):
        self.state = state
        self.parent = parent

# === Simula sequência de controles reais fora do CasADi (para checagem final) ===
def simulate_dynamics(state: State, A: np.ndarray, u_seq: list, m: float, J: np.ndarray, dt: float) -> State:
    x, v, q, w = state.x.copy(), state.v.copy(), state.q.copy(), state.w.copy()

    for u in u_seq:
        FM = A @ u
        F, M = FM[:3], FM[3:]

        # Transforma força para o mundo
        acc = (1 / m) * R.from_quat(q).apply(F)
        v += acc * dt
        x += v * dt

        # Atualiza orientação
        def Q(q):
            qx, qy, qz, qw = q
            return np.array([
                [ qw, -qz,  qy],
                [ qz,  qw, -qx],
                [-qy,  qx,  qw],
                [-qx, -qy, -qz],
            ])
        dq = 0.5 * Q(q) @ w
        q += dq * dt
        q /= np.linalg.norm(q)

        # Atualiza velocidade angular
        dw = np.linalg.inv(J) @ (M - np.cross(w, J @ w))
        w += dw * dt

    return State(x, v, q, w)

# === Planejador Lazy RRT com otimização CasADi ===
class LazyRRT_Casadi:
    def __init__(self, x_init, x_goal, A, J, m, max_iters=300, N=10, dt=0.1, goal_tolerance=1.0):
        self.x_init = x_init
        self.x_goal = x_goal
        self.A = A
        self.J = J
        self.m = m
        self.N = N
        self.dt = dt
        self.max_iters = max_iters
        self.goal_tolerance = goal_tolerance
        self.tree = [Node(x_init)]

    def sample(self):
        pos = np.random.uniform(0, 3, size=3)
        q = R.random().as_quat()
        return State(pos, np.zeros(3), q, np.zeros(3))

    def nearest(self, x_rand):
        return min(self.tree, key=lambda node: node.state.distance_to(x_rand))

    def try_connect(self, from_state, to_state):
        opti = ca.Opti()
        U = opti.variable(6, self.N)

        x = ca.MX(from_state.x)
        v = ca.MX(from_state.v)
        q = ca.MX(from_state.q)
        w = ca.MX(from_state.w)

        def Q(q):
            qx, qy, qz, qw = q[0], q[1], q[2], q[3]
            return ca.vertcat(
                ca.horzcat( qw, -qz,  qy),
                ca.horzcat( qz,  qw, -qx),
                ca.horzcat(-qy,  qx,  qw),
                ca.horzcat(-qx, -qy, -qz),
            )

        for k in range(self.N):
            u = U[:, k]
            FM = self.A @ u
            F = FM[:3]
            M = FM[3:]

            acc = (1 / self.m) * F  # Assume frame alinhado no CasADi
            v += acc * self.dt
            x += v * self.dt

            dq = 0.5 * Q(q) @ w
            q += dq * self.dt
            q = q / ca.norm_2(q)

            dw = ca.solve(self.J, (M - ca.cross(w, self.J @ w)))
            w += dw * self.dt

        cost = ca.sumsqr(x - to_state.x) + (1 - ca.dot(q, to_state.q))
        opti.minimize(cost)
        opti.subject_to(opti.bounded(-1, U, 1))

        opti.solver("ipopt", {"print_time": False}, {"print_level": 0})

        try:
            sol = opti.solve()
            U_opt = sol.value(U)
            u_seq = [U_opt[:, i] for i in range(self.N)]
            new_state = simulate_dynamics(from_state, self.A, u_seq, self.m, self.J, self.dt)
            if new_state.distance_to(to_state) < self.goal_tolerance:
                return new_state
        except:
            return None
        return None

    def plan(self):
        for _ in range(self.max_iters):
            x_rand = self.sample()
            x_nearest = self.nearest(x_rand)
            x_new = self.try_connect(x_nearest.state, x_rand)
            if x_new:
                new_node = Node(x_new, x_nearest)
                self.tree.append(new_node)
                if x_new.distance_to(self.x_goal) < self.goal_tolerance:
                    return self.reconstruct_path(new_node)
        return []

    def reconstruct_path(self, node):
        path = []
        while node:
            path.append(node.state)
            node = node.parent
        return path[::-1]

# === MAIN EXECUÇÃO ===
if __name__ == "__main__":
    J = np.load("J_matrix.npy")
    A = np.load("A_matrix.npy")
    m = np.load("mass.npy").item()

    x_init = State(x=np.array([0, 0, 0]), v=np.zeros(3), q=np.array([0, 0, 0, 1]), w=np.zeros(3))
    x_goal = State(x=np.array([2, 2, 2]), v=np.zeros(3), q=np.array([0, 0, 0, 1]), w=np.zeros(3))

    planner = LazyRRT_Casadi(x_init, x_goal, A, J, m)
    path = planner.plan()

    # Plot da trajetória (posição apenas)
    if path:
        xs = [s.x[0] for s in path]
        ys = [s.x[1] for s in path]
        zs = [s.x[2] for s in path]
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(xs, ys, zs, 'o-', label="Planned Path")
        ax.set_title("Lazy RRT with Dynamics (SE(3))")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        plt.legend()
        plt.tight_layout()
        plt.show()
    else:
        print("No path found.")
