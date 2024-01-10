def 导(f):
    import sympy as sp
    import math
    x = sp.Symbol('x')
    f_expr = eval(f)
    f_ = sp.diff(f_expr,x)
    return f_

def 画(f, x_lef=-5, x_rig=5):
    import numpy as np
    import sympy as sp
    import matplotlib.pyplot as plt

    x = sp.Symbol('x')
    f_np = sp.lambdify(x, f, 'numpy')
    x_vals = np.linspace(x_lef, x_rig, 100)
    y_vals = f_np(x_vals)

    plt.plot(x_vals, y_vals, label=f'{f}')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.axhline(y=0, linestyle='--', color='#FF0000')
    plt.axvline(x=0, linestyle='--', color='green')
    # plt.legend(loc='upper right')  # 调整图例位置
    # plt.tight_layout()  # 自动调整图像布局
    plt.show()
    print(f'Draw {f} !')

"""
x = sp.Symbol('x')
f = sp.cos(x)
画(f)
"""