import numpy as np

def quat_mul(q1, q2):
    """Quaternion multiply, q = q1 * q2, with [w, x, y, z]."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

def quat_conj(q):
    """Quaternion conjugate."""
    w, x, y, z = q
    return np.array([w, -x, -y, -z])

def quat_inv(q):
    """Quaternion inverse."""
    return quat_conj(q) / np.dot(q, q)

def quat_normalize(q):
    return q / np.linalg.norm(q)

def rotation_between_quaternions(q1, q2):
    """
    Find q_delta such that:
        q_delta * q1 = q2
    therefore:
        q_delta = q2 * inv(q1)
    """
    q1 = quat_normalize(np.asarray(q1, dtype=float))
    q2 = quat_normalize(np.asarray(q2, dtype=float))
    q_delta = quat_mul(q2, quat_inv(q1))
    return quat_normalize(q_delta)

# Example
q1 = [-0.707105,0.707108,0,0]
q2 = [0.70,-0.70,0,0]

q_delta = rotation_between_quaternions(q1, q2)
print("Rotation from q1 to q2:", q_delta)

# Check:
q2_check = quat_mul(q_delta, quat_normalize(q1))
print("q2 expected:", quat_normalize(q2))
print("q2 check:   ", quat_normalize(q2_check))