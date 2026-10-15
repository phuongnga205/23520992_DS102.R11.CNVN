import numpy as np

# 1. Dinh nghia ham so f(x, y) = x^4 + y^2 - 4*x - 2*y
def f(u):
    x, y = u[0], u[1]
    return x**4 + y**2 - 4*x - 2*y

# 2. Vector Gradient (dao ham bac 1)
def gradient(u):
    x, y = u[0], u[1]
    df_dx = 4*(x**3) - 4
    df_dy = 2*y - 2
    return np.array([df_dx, df_dy])

# 3. Ma tran Hessian (dao ham bac 2)
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

# 4. Thuat toan Newton-Raphson
def newton_raphson(u_init, max_iter=20, tol=1e-5):
    u = np.array(u_init, dtype=float)
    
    for i in range(max_iter):
        grad = gradient(u)
        H = hessian(u)
        
        # Kiem tra dieu kien dung (khi do lon gradient nho hon tol)
        if np.linalg.norm(grad) < tol:
            print(f"Hoi tu tai lan lap thu {i}")
            break
            
        # Giai he phuong trinh H * delta_u = -grad
        delta_u = np.linalg.solve(H, -grad)
        
        # Cap nhat vi tri
        u = u + delta_u
        
        print(f"Lan lap {i+1}: x = {u[0]:.4f}, y = {u[1]:.4f}, f(u) = {f(u):.6f}")
        
    return u

# --- CHAY THU CHUONG TRINH ---
# Chon diem bat dau u0 = [2.0, 3.0]
u_start = [2.0, 3.0]
u_min = newton_raphson(u_start)

print("\nDiem cuc tri tim duoc (x, y):", u_min)
print("Gia tri cuc tieu f(x, y):", round(f(u_min), 4))
