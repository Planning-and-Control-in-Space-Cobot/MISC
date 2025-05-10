import numpy as np
import fcl
import pyvista as pv
from scipy.spatial.transform import Rotation as R


def ellipsoid_signed_distance(center, rot_matrix, a, b, c, normal, closest_point):
    Q = np.diag([a**2, b**2, c**2])
    RQRT = rot_matrix @ Q @ rot_matrix.T
    extent = np.sqrt(normal.T @ RQRT @ normal)
    signed_distance = normal.T @ (center - closest_point) - extent
    return signed_distance


def main():
    # === Plane (thin box) ===
    normal = np.array([1.0, 0.0, 0.0])
    offset = 5.0
    box_thickness = 0.001
    box_width = 100.0
    box_height = 100.0
    box_geom = fcl.Box(box_thickness, box_width, box_height)
    box_position = np.array([offset + box_thickness / 2.0, 0.0, 0.0])
    box_transform = fcl.Transform(box_position)
    box_obj = fcl.CollisionObject(box_geom, box_transform)

    # === Ellipsoid ===
    a, b, c = 0.24, 0.24, 0.10
    ellipsoid_geom = fcl.Ellipsoid(a, b, c)
    center = np.array([3.0, 3.0, 3.0])
    euler_deg = [22, 90, 0]
    rotation = R.from_euler("xyz", np.radians(euler_deg))
    quat = rotation.as_quat()  # FCL
    rot_matrix = rotation.as_matrix()  # Support function
    ellipsoid_transform = fcl.Transform(quat, center)
    ellipsoid_obj = fcl.CollisionObject(ellipsoid_geom, ellipsoid_transform)

    # === FCL closest points ===
    request = fcl.DistanceRequest(enable_nearest_points=True)
    result = fcl.DistanceResult()

    try:
        distance = fcl.distance(ellipsoid_obj, box_obj, request, result)
        closest_ellipsoid = np.array(result.nearest_points[0])
        closest_box = np.array(result.nearest_points[1])
        direction = closest_ellipsoid - closest_box
        norm = np.linalg.norm(direction)
        if norm == 0:
            print("⚠️ Closest points coincide, normal undefined.")
            return
        direction_normal = direction / norm

        print("=== FCL Computed Distance ===")
        print(f"Minimum distance: {distance:.6f}")
        print(f"Closest point on ellipsoid: {closest_ellipsoid}")
        print(f"Closest point on box: {closest_box}")
        print(f"Direction vector (normal): {direction_normal}")

        # === Support Function-Based Distance ===
        signed_dist = ellipsoid_signed_distance(
            center=center,
            rot_matrix=rot_matrix,
            a=a,
            b=b,
            c=c,
            normal=direction_normal,
            closest_point=closest_box,
        )

        print("\n=== Support Function-Based Distance ===")
        print(f"Signed distance: {signed_dist:.6f}")

        # === Visualization ===
        plotter = pv.Plotter()
        plotter.add_axes()
        plotter.set_background("white")

        # Box (plane)
        box_mesh = pv.Cube(
            center=box_position,
            x_length=box_thickness,
            y_length=box_width,
            z_length=box_height,
        )
        plotter.add_mesh(box_mesh, color="gray", opacity=0.5, label="Plane")

        # Ellipsoid mesh (correct rotation and translation)
        ellipsoid = pv.ParametricEllipsoid(a, b, c)
        T = np.eye(4)
        T[:3, :3] = rot_matrix
        T[:3, 3] = center
        ellipsoid.transform(T)
        plotter.add_mesh(ellipsoid, color="limegreen", opacity=0.9, label="Ellipsoid")

        # Closest points
        sphere1 = pv.Sphere(radius=0.03, center=closest_ellipsoid)
        sphere2 = pv.Sphere(radius=0.03, center=closest_box)
        plotter.add_mesh(sphere1, color="blue", label="Closest Ellipsoid Point")
        plotter.add_mesh(sphere2, color="red", label="Closest Box Point")

        # Line between closest points
        line = pv.Line(closest_ellipsoid, closest_box)
        plotter.add_mesh(line, color="black", line_width=3)

        plotter.show()

    except Exception as e:
        print(f"Error during distance computation: {e}")


if __name__ == "__main__":
    main()
