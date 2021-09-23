# !
from matplotlib import colors
import numpy as np
import matplotlib.pyplot as plt


def calCurve(theta=np.pi / 6, v0=1000, k=0.3):
    eps = 0.000000001

    t_interval = 0.0001
    # t_interval = 10
    p = np.array([0, 0], dtype=np.float32)
    v = np.array([v0 * np.cos(theta), v0 * np.sin(theta)], dtype=np.float32)
    m = 30
    f = -k * v ** 2
    # f=np.array([0, 0])
    g = np.array([0, -9.8], dtype=np.float32)

    fig = plt.figure()
    xs, ys = [], []

    while p[1] >= -eps:
        # print(f, g)
        a = f / m + g
        v = v + a * t_interval
        p = p + v * t_interval
        xs.append(p[0])
        ys.append(p[1])

        # print(p[0], p[1])
        f = -k * v ** 2
    return xs, ys


if __name__ == "__main__":
    fig, ax = plt.subplots()
    xs1, ys1 = calCurve(theta=np.pi / 6)
    xs2, ys2 = calCurve(theta=np.pi / 4)
    xs3, ys3 = calCurve(theta=np.pi / 3)
    ax.plot(xs1, ys1, label="Parabola curve when theta=pi/6", color="r")
    ax.plot(xs2, ys2, label="Parabola curve when theta=pi/4", color="b")
    ax.plot(xs3, ys3, label="Parabola curve when theta=pi/3", color="g")
    ax.grid(axis="y")
    ax.legend(
        [
            "Parabola curve when theta=pi/6",
            "Parabola curve when theta=pi/4",
            "Parabola curve when theta=pi/3",
        ]
    )
    ax.set_xlabel("x axis, (m)")
    ax.set_ylabel("y axis, (m)")
    fig.savefig("Parabola curve.png")

    max_xs = 0
    # print(np.arange(2*np.pi/360, np.pi/2, 2*np.pi/360))
    for theta in np.arange(2*np.pi/360, np.pi/2, 2*np.pi/360):
        xs, ys = calCurve(theta=theta)
        max_xs = max(xs[-1], max_xs)
    print(f"最大的轨迹长度是{max_xs}")

