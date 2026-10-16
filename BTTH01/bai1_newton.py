import numpy as np

def f(u):
    x, y = u[0], u[1]
    return x**4 + y**2 - 4*x - 2*y

def gradient(u):
    x, y = u[0], u[1]
    df_dx = 4*(x**3) - 4
    df_dy = 2*y - 2
    return np.array([df_dx, df_dy])

def hessian(u):
    x, y = u[0], u[1]
    d2f_dx2 = 12*(x**2)
    d2f_dxdy = 0.0
    d2f_dydx = 0.0
    d2f_dy2 = 2.0
    
    return np.array([
        [d2f_dx2, d2f_dxdy],
        [d2f_dydx, d2f_dy2]
    ])

def newton_raphson(u_init, max_iter=20, tol=1e-5):
    u = np.array(u_init, dtype=float)
    
    for i in range(max_iter):
        grad = gradient(u)
        H = hessian(u)
        
        if np.linalg.norm(grad) < tol:
            print(f"Hoi tu tai lan lap thu {i}")
            break
            
        delta_u = np.linalg.solve(H, -grad)
        
        u = u + delta_u
        
        print(f"Lan lap {i+1}: x = {u[0]:.4f}, y = {u[1]:.4f}, f(u) = {f(u):.6f}")
        
    return u

u_start = [2.0, 3.0]
u_min = newton_raphson(u_start)

print("\nDiem cuc tri tim duoc (x, y):", u_min)
print("Gia tri cuc tieu f(x, y):", round(f(u_min), 4))
