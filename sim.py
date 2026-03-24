import numpy as np
from vedo import *

if not os.path.exists("robot/"):
    os.system("wget -O robot.zip \"https://www.dropbox.com/scl/fi/uewvrcempf2wf2jp7bcb8/robot.zip?rlkey=7uwz1ne94hxyinub8x16y93em&dl=1\"")
    os.system("unzip robot.zip")
    os.system("rm robot.zip")


class RobotArm:
    """
    Robot arm with:
      - forward kinematics
      - Jacobian-based IK
      - persistent vedo meshes updated in-place
    """

    def __init__(self, partLengths, parts, arm_location):
        self.arm_location = np.array(arm_location, dtype=float)
        self.L1, self.L2, self.L3, self.L4 = partLengths
        self.source_parts = parts

        self.delta_phi        = 0.5    # larger delta avoids near-zero Jacobian columns
        self.target           = np.array([4.5, 0.0, 0.3], dtype=float)
        self.target_tolerance = 0.3
        self.target_lambda    = 0.05
        self.convergence      = 0.02
        self.iteration_limit  = 1000
        self.ik_damping       = 0.05   # Levenberg-Marquardt damping
        self.ik_step_scale    = 5.0    # step size per frame

        self.meshes = None
        self.initialize_meshes()

    def initialize_meshes(self):
        self.meshes = [p.clone() for p in self.source_parts]
        self._source_pts = [np.array(p.points) for p in self.source_parts]

    def RotationMatrix(self, theta, axis_name):
        c = np.cos(theta * np.pi / 180.0)
        s = np.sin(theta * np.pi / 180.0)
        if axis_name == "x":
            return np.array([[1,0,0],[0,c,-s],[0,s,c]])
        elif axis_name == "y":
            return np.array([[c,0,s],[0,1,0],[-s,0,c]])
        elif axis_name == "z":
            return np.array([[c,-s,0],[s,c,0],[0,0,1]])
        raise ValueError(f"Unknown axis_name: {axis_name}")

    def createCoordinateFrameMesh(self):
        shaft_radius = 0.05
        head_radius  = 0.10
        alpha        = 1.0
        unit         = 30
        x_axisArrow = Arrow((0,0,0),(unit,0,0), shaft_radius=shaft_radius,
                            head_radius=head_radius, res=12, c="red", alpha=alpha)
        y_axisArrow = Arrow((0,0,0),(0,unit,0), shaft_radius=shaft_radius,
                            head_radius=head_radius, res=12, c="green", alpha=alpha)
        z_axisArrow = Arrow((0,0,0),(0,0,unit), shaft_radius=shaft_radius,
                            head_radius=head_radius, res=12, c="blue", alpha=alpha)
        originDot = Sphere(pos=[0,0,0], c="black", r=0.10*unit)
        return x_axisArrow + y_axisArrow + z_axisArrow + originDot

    def getLocalFrameMatrix(self, R_ij, t_ij):
        t_ij = np.array(t_ij).reshape(3, 1)
        return np.block([
            [R_ij, t_ij],
            [np.zeros((1, 3)), np.array([[1]])]
        ])

    def forward_kinematics(self, Phi):
        phi1 = Phi[0]
        R_00 = self.RotationMatrix(0, axis_name="z")
        t_00 = self.arm_location.reshape(3)
        T_00 = self.getLocalFrameMatrix(R_00, t_00)

        R_01 = self.RotationMatrix(phi1, axis_name="z")
        t_01 = np.array([0.0, 0.0, 0.0])
        T_01 = T_00 @ self.getLocalFrameMatrix(R_01, t_01)

        phi2 = Phi[1]
        R_12 = self.RotationMatrix(phi2, axis_name="y")
        t_12 = np.array([0.0, 0.0, self.L1])
        T_02 = T_01 @ self.getLocalFrameMatrix(R_12, t_12)

        phi3 = Phi[2]
        R_23 = self.RotationMatrix(phi3, axis_name="y")
        t_23 = np.array([0.0, 0.0, self.L2])
        T_03 = T_02 @ self.getLocalFrameMatrix(R_23, t_23)

        phi4 = Phi[3]
        R_34 = self.RotationMatrix(phi4, axis_name="y")
        t_34 = np.array([0.0, 0.0, self.L3])
        T_04 = T_03 @ self.getLocalFrameMatrix(R_34, t_34)

        e = T_04[0:3, -1]
        return T_00, T_01, T_02, T_03, T_04, e

    def get_pose_transforms(self, Phi):
        T_00, T_01, T_02, T_03, T_04, e = self.forward_kinematics(Phi)
        return [T_00, T_01, T_02, T_03, T_04], e

    def update_pose(self, Phi):
        transforms, e = self.get_pose_transforms(Phi)
        for i, mesh in enumerate(self.meshes):
            pts = self._source_pts[i]
            T   = transforms[i]
            R, t = T[:3, :3], T[:3, 3]
            mesh.points = pts @ R.T + t
        return self.meshes, e

    def jacobian_matrix(self, phi):
        # 3×4 Jacobian — all four joints included.
        # Joint 1 (Z-rotation) controls Y reachability; omitting it
        # makes the Y column identically zero and breaks IK.
        step = self.delta_phi
        _, _, _, _, _, e = self.forward_kinematics(phi)
        cols = []
        for i in range(4):
            dp = np.zeros(4); dp[i] = step
            _, _, _, _, _, ei = self.forward_kinematics(phi + dp)
            cols.append((ei - e) / step)
        return np.column_stack(cols)   # shape (3, 4)

    def ik_step(self, phi):
        """
        Three damped-least-squares substeps per timer frame.
        Multiple small clamped substeps converge ~3x faster than one large step
        while staying stable — a single unclamped step overshoots and oscillates.
        """
        MAX_DEG = 5.0   # max degrees any joint moves per substep
        lam     = 0.05  # Levenberg-Marquardt damping

        phi = phi.copy()
        for _ in range(3):
            _, _, _, _, _, e = self.forward_kinematics(phi)
            error = self.target - e
            if np.linalg.norm(error) < self.target_tolerance:
                break
            J   = self.jacobian_matrix(phi)           # (3, 4)
            A   = J @ J.T + lam**2 * np.eye(3)
            raw = 2.0 * (J.T @ np.linalg.solve(A, error))
            phi += np.clip(raw, -MAX_DEG, MAX_DEG)
        return phi

    def inverse_kinematics_newton(self, initial_phi, record_every=20, motion_threshold=10):
        phi = np.array(initial_phi, dtype=float).copy()
        _, _, _, _, _, e = self.forward_kinematics(phi)

        target          = np.array(self.target, dtype=float)
        recorder        = [phi.copy()]
        iteration       = 0
        e_accumulate    = 0.0
        lam             = 0.05   # damping — same as ik_step

        while np.linalg.norm(target - e) > self.target_tolerance:
            iteration += 1

            J   = self.jacobian_matrix(phi)           # (3, 4) full Jacobian
            A   = J @ J.T + lam**2 * np.eye(3)
            phi_delta = J.T @ np.linalg.solve(A, self.target_lambda * (target - e))
            phi += phi_delta

            e_previous = e.copy()
            _, _, _, _, _, e = self.forward_kinematics(phi)
            e_accumulate += np.linalg.norm(e_previous - e)

            if iteration % record_every == 0 or e_accumulate > motion_threshold:
                recorder.append(phi.copy())
                e_accumulate = 0.0

            if np.linalg.norm(e_previous - e) < self.convergence:
                break
            if iteration > self.iteration_limit:
                break

        if not np.allclose(recorder[-1], phi):
            recorder.append(phi.copy())

        return np.array(recorder)


# ================================================================
# BUILDING GEOMETRY
# ================================================================
FLOOR_HEIGHTS    = [0, 3, 6]
FLOOR_SIZE       = 10
FLOOR_THICKNESS  = 0.2
WALL_HEIGHT      = 2.5
WALL_THICKNESS   = 0.2

RAMPS_DEF = [
    dict(A=np.array([0.0,  3.0]), B=np.array([3.0,  3.0]), z0=3, z1=0, r=1.5),
    dict(A=np.array([0.0, -3.0]), B=np.array([3.0, -3.0]), z0=6, z1=3, r=1.2),
]

EXITS_XY = [np.array([4.5, 0.0]), np.array([0.0, -4.5])]

floors = []
for z in FLOOR_HEIGHTS:
    floors.append(Box(pos=(0,0,z), length=FLOOR_SIZE, width=FLOOR_SIZE,
                      height=FLOOR_THICKNESS).color("darkgray").alpha(0.6).lighting("off"))

def generate_ramp_mesh(ramp, res=(50, 20)):
    A, B, z0, z1, r = ramp["A"], ramp["B"], ramp["z0"], ramp["z1"], ramp["r"]
    v = B - A; length = np.linalg.norm(v); v_hat = v / length
    n_hat = np.array([-v_hat[1], v_hat[0]])
    ns, nw = res
    pts = []
    for s in np.linspace(0, length, ns):
        for w in np.linspace(-r, r, nw):
            xy = A + s * v_hat + w * n_hat
            pts.append([xy[0], xy[1], z0 + (s / length) * (z1 - z0)])
    pts   = np.array(pts)
    faces = []
    for i in range(ns - 1):
        for j in range(nw - 1):
            a = i * nw + j
            faces.append([a, a+1, a+nw+1, a+nw])
    return Mesh([pts, faces]).color("tan").alpha(0.5).lighting("off")

ramp_meshes = [generate_ramp_mesh(r) for r in RAMPS_DEF]

walls = []
half  = FLOOR_SIZE / 2
wzo   = FLOOR_THICKNESS / 2 + WALL_HEIGHT / 2
for z in FLOOR_HEIGHTS:
    wz = z + wzo
    walls += [
        Box(pos=( half, 0, wz), length=WALL_THICKNESS, width=FLOOR_SIZE,     height=WALL_HEIGHT),
        Box(pos=(-half, 0, wz), length=WALL_THICKNESS, width=FLOOR_SIZE,     height=WALL_HEIGHT),
        Box(pos=(0,  half, wz), length=FLOOR_SIZE,     width=WALL_THICKNESS, height=WALL_HEIGHT),
        Box(pos=(0, -half, wz), length=FLOOR_SIZE,     width=WALL_THICKNESS, height=WALL_HEIGHT),
    ]
for w in walls:
    w.color("lightgray").alpha(0.08).lighting("off")

def make_wall(x, y, l, w, z):
    return Box(pos=(x, y, z + WALL_HEIGHT/2), length=l, width=w,
               height=WALL_HEIGHT).color("silver").lighting("off")

interior_walls = [
    make_wall(-0.5, -2.75, WALL_THICKNESS, 2.5, FLOOR_HEIGHTS[0]),
    make_wall(-3.25,  1.0, 2.5, WALL_THICKNESS, FLOOR_HEIGHTS[0]),
    make_wall(-3.5,   3.25, WALL_THICKNESS, 2.5, FLOOR_HEIGHTS[0]),
    make_wall( 0.5,  -3.25, WALL_THICKNESS, 1.5, FLOOR_HEIGHTS[0]),
    make_wall(-1.8,   1.0, WALL_THICKNESS, 5.0, FLOOR_HEIGHTS[1]),
    make_wall(-3.15,  3.0, 2.7, WALL_THICKNESS, FLOOR_HEIGHTS[1]),
    make_wall(-3.15,  0.5, 2.7, WALL_THICKNESS, FLOOR_HEIGHTS[1]),
    make_wall(-3.15, -1.8, 2.7, WALL_THICKNESS, FLOOR_HEIGHTS[1]),
    make_wall(-2.5,  -3.35, WALL_THICKNESS, 2.3, FLOOR_HEIGHTS[1]),
    make_wall( 0.0,   0.8, 8.0, WALL_THICKNESS, FLOOR_HEIGHTS[2]),
    make_wall( 0.8,   1.75, WALL_THICKNESS, 4.5, FLOOR_HEIGHTS[2]),
    make_wall(-1.85,  3.0, 4.3, WALL_THICKNESS, FLOOR_HEIGHTS[2]),
    make_wall(-1.85, -1.5, 4.3, WALL_THICKNESS, FLOOR_HEIGHTS[2]),
    make_wall( 2.9,   3.0, 3.2, WALL_THICKNESS, FLOOR_HEIGHTS[2]),
]

wall_bounds  = np.array([w.bounds() for w in interior_walls])
wall_floor_z = np.array([0]*4 + [3]*5 + [6]*5, dtype=float)

eh = FLOOR_HEIGHTS[0] + FLOOR_THICKNESS/2 + 0.01
exits_vis = [
    Box(pos=(4.5,  0, eh), length=1.0, width=2.0, height=0.05).color("green").alpha(0.9),
    Box(pos=(0, -4.5, eh), length=2.0, width=1.0, height=0.05).color("green").alpha(0.9),
]


# ================================================================
# HELPERS
# ================================================================
def floor_of(z):
    if z >= 4.5:
        return 6
    elif z >= 1.5:
        return 3
    else:
        return 0

def on_ramp(xy, z):
    if z < 0.1:
        return None, None
    for ramp in RAMPS_DEF:
        zlo = min(ramp["z0"], ramp["z1"])
        zhi = max(ramp["z0"], ramp["z1"])
        if z < zlo - 0.6 or z > zhi + 0.6:
            continue
        v  = ramp["B"] - ramp["A"]
        t  = float(np.clip(np.dot(xy - ramp["A"], v) / np.dot(v, v), 0, 1))
        closest = ramp["A"] + t * v
        if np.linalg.norm(xy - closest) <= ramp["r"]:
            return ramp, t
    return None, None

MARGIN = 0.1

def slide_move(px, py, dx, dy, floor_z):
    wb = wall_bounds[wall_floor_z == floor_z]

    def hit(x, y):
        return bool(np.any(
            (wb[:, 0] - MARGIN < x) & (x < wb[:, 1] + MARGIN) &
            (wb[:, 2] - MARGIN < y) & (y < wb[:, 3] + MARGIN)
        ))

    nx, ny = px + dx, py + dy
    if   not hit(nx, ny): return nx, ny
    elif not hit(nx, py): return nx, py
    elif not hit(px, ny): return px, ny
    else:                 return px, py


# ================================================================
# PARTICLES
# ================================================================
NUM_PARTICLES = 20
rng = np.random.default_rng()

def spawn_ok(x, y, floor_z):
    wb = wall_bounds[wall_floor_z == floor_z]
    if len(wb) == 0:
        return True
    return not bool(np.any(
        (wb[:, 0] - MARGIN < x) & (x < wb[:, 1] + MARGIN) &
        (wb[:, 2] - MARGIN < y) & (y < wb[:, 3] + MARGIN)
    ))

particles = []
for _ in range(NUM_PARTICLES):
    fz = float(rng.choice(FLOOR_HEIGHTS))
    while True:
        x = rng.uniform(-4, 4)
        y = rng.uniform(-4, 4)
        if spawn_ok(x, y, fz):
            break
    particles.append(np.array([x, y, fz + 0.25]))

exited = [False] * NUM_PARTICLES

STEP     = 0.04
EXIT_TOL = 0.45


# ================================================================
# MOVEMENT
# ================================================================
def get_target_xy(p):
    xy = p[:2]
    z  = float(p[2])

    ramp, t = on_ramp(xy, z)
    if ramp is not None:
        v_hat = ramp["B"] - ramp["A"]
        v_hat = v_hat / (np.linalg.norm(v_hat) + 1e-9)
        overshoot = ramp["B"] + v_hat * (ramp["r"] + 0.5)
        return overshoot

    cur_floor = floor_of(z)

    if cur_floor == 0:
        dists = [np.linalg.norm(xy - e) for e in EXITS_XY]
        return EXITS_XY[int(np.argmin(dists))]

    for r in RAMPS_DEF:
        if r["z0"] == cur_floor:
            return r["A"].astype(float)

    return EXITS_XY[0]


def move_particle(p):
    xy = p[:2].copy()
    z  = float(p[2])

    target_xy = get_target_xy(p)
    diff = target_xy - xy
    dist = np.linalg.norm(diff)
    if dist < 1e-6:
        return p

    step_xy = diff / dist * min(STEP, dist)

    ramp, t = on_ramp(xy, z)
    if ramp is not None:
        nx = float(np.clip(xy[0] + step_xy[0], -4.9, 4.9))
        ny = float(np.clip(xy[1] + step_xy[1], -4.9, 4.9))
    else:
        nx, ny = slide_move(xy[0], xy[1], step_xy[0], step_xy[1], float(floor_of(z)))
        nx = float(np.clip(nx, -4.9, 4.9))
        ny = float(np.clip(ny, -4.9, 4.9))

    new_ramp, new_t = on_ramp(np.array([nx, ny]), z)
    if new_ramp is not None:
        new_z = new_ramp["z0"] + new_t * (new_ramp["z1"] - new_ramp["z0"])
    else:
        new_z = float(floor_of(z))

    return np.array([nx, ny, new_z])


def move_particle_with_robot(p, ee_xy):
    new_p = move_particle(p)

    diff = new_p[:2] - ee_xy
    dist = np.linalg.norm(diff)
    repulsion_radius = 0.5
    if 1e-6 < dist < repulsion_radius:
        repulse = (repulsion_radius - dist) * (diff / dist)
        new_p[:2] += repulse

    nx, ny = slide_move(new_p[0], new_p[1], 0, 0, float(floor_of(new_p[2])))
    new_p[:2] = [nx, ny]

    new_ramp, new_t = on_ramp(new_p[:2], new_p[2])
    if new_ramp is not None:
        new_p[2] = new_ramp["z0"] + new_t * (new_ramp["z1"] - new_ramp["z0"])
    else:
        new_p[2] = float(floor_of(new_p[2]))

    return new_p


def get_particles_near_exit(particles, exited, exit_xy, radius=2.0):
    near = []
    for i, p in enumerate(particles):
        if exited[i]:
            continue
        if floor_of(p[2]) != 0:      # only consider particles on the ground floor
            continue
        if np.linalg.norm(p[:2] - exit_xy) < radius:
            near.append(p[:2])
    return np.array(near)


def predict_cluster(positions, prev_center=None, alpha=0.5):
    if len(positions) == 0:
        return np.array([4.5, 0.0, 0.3])
    center = positions.mean(axis=0)
    if prev_center is None:
        return np.array([center[0], center[1], 0.3])
    velocity = center - prev_center
    predicted = center + alpha * velocity
    return np.array([predicted[0], predicted[1], 0.3])

def kmeans_2clusters(points, max_iters=10):
    """
    Very lightweight k-means (k=2).
    Returns: centers (2x2), labels (N,)
    """
    if len(points) < 2:
        return points, np.zeros(len(points), dtype=int)

    # Initialize centers randomly from points
    rng = np.random.default_rng()
    centers = points[rng.choice(len(points), size=2, replace=False)]

    for _ in range(max_iters):
        # Assign points to nearest center
        dists = np.linalg.norm(points[:, None, :] - centers[None, :, :], axis=2)
        labels = np.argmin(dists, axis=1)

        # Recompute centers
        new_centers = []
        for k in range(2):
            cluster_pts = points[labels == k]
            if len(cluster_pts) == 0:
                new_centers.append(centers[k])  # keep old if empty
            else:
                new_centers.append(cluster_pts.mean(axis=0))
        new_centers = np.array(new_centers)

        # Stop if converged
        if np.allclose(new_centers, centers):
            break
        centers = new_centers

    return centers, labels

def get_dominant_cluster(points):
    """
    Runs k-means and returns the largest cluster.
    """
    if len(points) == 0:
        return None, None

    if len(points) < 3:
        return points, points.mean(axis=0)

    centers, labels = kmeans_2clusters(points)

    # Count cluster sizes
    counts = [np.sum(labels == k) for k in range(2)]
    dominant_k = int(np.argmax(counts))

    cluster_pts = points[labels == dominant_k]
    cluster_center = centers[dominant_k]

    return cluster_pts, cluster_center


# ================================================================
# VEDO SETUP
# ================================================================
plotter = Plotter()
plotter.add(*floors, *walls, *interior_walls, *exits_vis, *ramp_meshes)
spheres = [Sphere(pos=p, r=0.15).color("red5") for p in particles]
plotter.add(*spheres)

scale_factor = 0.0050
robot_L = np.array([105, 217, 416, 0]) * scale_factor

# ── Robot 1 — guards exit 0 at [4.5, 0] ──────────────────────────
Base1    = load("./robot/Base.stl").scale(scale_factor).color("blue5")
BaseRot1 = load("./robot/BaseRot.stl").scale(scale_factor).color("lightblue")
Humerus1 = load("./robot/Humerus.stl").scale(scale_factor).color("gray5")
Radius1  = load("./robot/Radius.stl").scale(scale_factor).color("red5")

robot1     = RobotArm(robot_L, [Base1, BaseRot1, Humerus1, Radius1], np.array([4.5, -1.5, 0.0]))
robot1_phi = np.array([0.0, 0.0, 0.0, 0.0])
robot1.update_pose(robot1_phi)
_, _, _, _, _, ee1_pos = robot1.forward_kinematics(robot1_phi)
ee1_xy = ee1_pos[:2].copy()
plotter.add(*robot1.meshes)

# ── Robot 2 — guards exit 1 at [0, -4.5] ─────────────────────────
Base2    = load("./robot/Base.stl").scale(scale_factor).color("purple")
BaseRot2 = load("./robot/BaseRot.stl").scale(scale_factor).color("violet")
Humerus2 = load("./robot/Humerus.stl").scale(scale_factor).color("gray5")
Radius2  = load("./robot/Radius.stl").scale(scale_factor).color("orange")

robot2     = RobotArm(robot_L, [Base2, BaseRot2, Humerus2, Radius2], np.array([-1.5, -4.5, 0.0]))
robot2_phi = np.array([0.0, 0.0, 0.0, 0.0])
robot2.target = np.array([0.0, -4.5, 0.3])   # start aimed at exit 2
robot2.update_pose(robot2_phi)
_, _, _, _, _, ee2_pos = robot2.forward_kinematics(robot2_phi)
ee2_xy = ee2_pos[:2].copy()
plotter.add(*robot2.meshes)

# ── Visualisation spheres: blue/yellow for robot1, cyan/magenta for robot2
cluster_sphere1 = Sphere(r=0.25).color("blue")
pred_sphere1    = Sphere(r=0.25).color("yellow")
cluster_sphere2 = Sphere(r=0.25).color("cyan")
pred_sphere2    = Sphere(r=0.25).color("magenta")
plotter.add(cluster_sphere1, pred_sphere1, cluster_sphere2, pred_sphere2)

prev_cluster_center1 = None
prev_cluster_center2 = None
tid = None


# ================================================================
# TIMER
# ================================================================
def on_timer(event):
    global robot1_phi, robot2_phi, tid
    global ee1_xy, ee2_xy
    global prev_cluster_center1, prev_cluster_center2

    # 1. Move particles — repelled by both end-effectors
    all_done = True
    for i, p in enumerate(particles):
        if exited[i]:
            continue
        new_p = move_particle_with_robot(p.copy(), ee1_xy)
        new_p = move_particle_with_robot(new_p,    ee2_xy)
        particles[i] = new_p
        spheres[i].pos(new_p)
        if floor_of(new_p[2]) == 0 and any(np.linalg.norm(new_p[:2] - ex) < EXIT_TOL for ex in EXITS_XY):
            exited[i] = True
            spheres[i].color("green").alpha(0.3)
        if not exited[i]:
            all_done = False

    near1 = get_particles_near_exit(particles, exited, EXITS_XY[0])

    if len(near1) > 0:
        cluster_pts1, cluster_center1 = get_dominant_cluster(near1)

        predicted1 = predict_cluster(cluster_pts1, prev_cluster_center1)
        prev_cluster_center1 = cluster_center1.copy()

        robot1.target = predicted1.copy()
    else:
        predicted1 = robot1.target.copy()
        cluster_center1 = EXITS_XY[0]

    near2 = get_particles_near_exit(particles, exited, EXITS_XY[1])

    if len(near2) > 0:
        cluster_pts2, cluster_center2 = get_dominant_cluster(near2)

        predicted2 = predict_cluster(cluster_pts2, prev_cluster_center2)
        prev_cluster_center2 = cluster_center2.copy()

        robot2.target = predicted2.copy()
    else:
        predicted2 = robot2.target.copy()
        cluster_center2 = EXITS_XY[1]

    # 3. Step both robots
    robot1_phi = robot1.ik_step(robot1_phi)
    robot1.update_pose(robot1_phi)
    _, _, _, _, _, ee1_pos = robot1.forward_kinematics(robot1_phi)
    ee1_xy = ee1_pos[:2].copy()

    robot2_phi = robot2.ik_step(robot2_phi)
    robot2.update_pose(robot2_phi)
    _, _, _, _, _, ee2_pos = robot2.forward_kinematics(robot2_phi)
    ee2_xy = ee2_pos[:2].copy()

    # 4. Update visualisation spheres
    cluster_sphere1.pos([cluster_center1[0], cluster_center1[1], 0.3])
    pred_sphere1.pos(predicted1)
    cluster_sphere2.pos([cluster_center2[0], cluster_center2[1], 0.3])
    pred_sphere2.pos(predicted2)

    # 5. Render
    plotter.render()

    if all_done:
        print("Simulation complete.")
        plotter.timer_callback("stop", tid)


tid = plotter.timer_callback("create", dt=20)
plotter.add_callback("timer", on_timer)
plotter.show(axes=1, interactive=True, title="Evacuation Simulation")