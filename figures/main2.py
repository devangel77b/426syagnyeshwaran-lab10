import numpy as np
import matplotlib.pyplot as plt
    
Nx, Ny = 60, 60
iters = 8000
tol = 1e-6
    
def harmonic_mean(a, b, eps=1e-12):
    return (2.0 * a * b) / (a + b + eps)
    
def compute_electric_field(V, hx=1.0, hy=1.0):
    dVdi, dVdj = np.gradient(V, hx, hy, edge_order=2)
    Ex = -dVdj 
    Ey = -dVdi
    return Ex, Ey
    
def solve_fd_conductance(with_flaw=False, omega=1.8):
    sigma = np.ones((Nx, Ny), dtype=float)
    if with_flaw:
        sigma[10:50, 30:32] = 1e-9
        
    V = np.zeros((Nx, Ny), dtype=float)
    for j in range(Ny):
        V[:, j] = 1.0 + (0.0 - 1.0) * (j / (Ny - 1))
    
    fixed = np.zeros((Nx, Ny), dtype=bool)
    fixed[:, 0] = True
    fixed[:, -1] = True
    
    def apply_bc(V):
        V[:, 0] = 1.0  
        V[:, -1] = 0.0
        V[0, :] = V[1, :]      
        V[-1, :] = V[-2, :]    
    
    apply_bc(V)
    
    for _ in range(iters):
        max_delta = 0.0
        for i in range(1, Nx - 1):
            for j in range(1, Ny - 1):
                if fixed[i, j]:
                    continue
    
                ip, im = i + 1, i - 1
                jp, jm = j + 1, j - 1
    
                sxp = harmonic_mean(sigma[i, j], sigma[ip, j])
                sxm = harmonic_mean(sigma[i, j], sigma[im, j])
                syp = harmonic_mean(sigma[i, j], sigma[i, jp])
                sym = harmonic_mean(sigma[i, j], sigma[i, jm])
    
                A = sxp + sxm + syp + sym
                if A < 1e-20:
                    continue

                # The next line implements relaxation or
                # lazy Gauss-Seidel
                V_ideal = (sxp * V[ip, j] + sxm * V[im, j] +
                           syp * V[i, jp] + sym * V[i, jm]) / A

                # The next line implements successive over-relaxation
                V_old = V[i, j]
                V[i, j] = (1.0 - omega) * V_old + omega * V_ideal
    
                delta = abs(V[i, j] - V_old)
                if delta > max_delta:
                    max_delta = delta
    
        apply_bc(V)
        if max_delta < tol:
            break
    
    return V, sigma
    
def add_arrowheads(ax, stream, color="white", lw=1.0):
    segs = stream.lines.get_segments()
    if len(segs) == 0:
        return
    
    k = 12  
    for idx in range(0, len(segs), k):
        seg = segs[idx]
        (x0, y0), (x1, y1) = seg[0], seg[1]
        ax.annotate(
            "",
            xy=(x1, y1),
            xytext=(x0, y0),
            arrowprops=dict(arrowstyle="->", color=color,
                            linewidth=lw, shrinkA=0, shrinkB=0),
        )
    
# Run
V_clean, sigma_clean = solve_fd_conductance(with_flaw=False)
V_flaw, sigma_flaw = solve_fd_conductance(with_flaw=True)

V_diff = V_flaw - V_clean
    
Ex_clean, Ey_clean = compute_electric_field(V_clean)
Ex_flaw, Ey_flaw = compute_electric_field(V_flaw)
    
# PLOTTING (four figures)
    
x = np.arange(Nx)   
y = np.arange(Ny)   
X, Y = np.meshgrid(x, y)
    
# No flaw (V + E streamlines)
fig,ax = plt.subplots(figsize=(5,4), constrained_layout=True, dpi=1200)
im0 = ax.imshow(V_clean, origin="lower", cmap="viridis", vmin=0, vmax=1, extent=[0,Nx-1,0,Ny-1])
s0 = ax.streamplot(
    X, Y, Ex_clean, Ey_clean,
    density=0.6,
    linewidth=0.9,
    color=(1, 1, 1, 0.75),
    arrowstyle='-'   
)
add_arrowheads(ax, s0, color=(1, 1, 1, 0.9), lw=1.0)
ax.set_xlabel("x (nodes)")
ax.set_ylabel("y (nodes)")
cb0=plt.colorbar(im0, ax=ax)
cb0.set_label("electric potential (V)")
fig.savefig("potential-noflaw.png",dpi=1200)
plt.close(fig)
    
# With flaw (V + E streamlines)
fig,ax = plt.subplots(figsize=(5,4), constrained_layout=True, dpi=1200)
im1 = ax.imshow(V_flaw, origin="lower", cmap="viridis", vmin=0, vmax=1,extent=[0,Nx-1,0,Ny-1])
s1 = ax.streamplot(
    X, Y, Ex_flaw, Ey_flaw,
    density=0.6,
    linewidth=0.9,
    color=(1, 1, 1, 0.75),
    arrowstyle='-'
)

add_arrowheads(ax, s1, color=(1, 1, 1, 0.9), lw=1.0)
ax.set_xlabel("x (nodes)")
ax.set_ylabel("y (nodes)")
cb1=plt.colorbar(im1, ax=ax)
cb1.set_label("electric potential (V)")
fig.savefig("potential-flaw.png",dpi=1200)
plt.close(fig)
    
# Difference signal
fig,ax = plt.subplots(figsize=(5,4), constrained_layout=True, dpi=1200)    
dvmax = np.max(np.abs(V_diff))
im2 = ax.imshow(V_diff, origin="lower", cmap="seismic", vmin=-dvmax, vmax=dvmax)
ax.set_xlabel("x (nodes)")
ax.set_ylabel("y (nodes)")
cb2=plt.colorbar(im2, ax=ax)
cb2.set_label("electric potential (V)")
fig.savefig("potential-difference.png",dpi=1200)
plt.close(fig)

# Conductivity map
fig,ax = plt.subplots(figsize=(5,4), constrained_layout=True, dpi=1200)
im3 = ax.imshow(sigma_flaw, origin="lower", cmap="gray_r")
ax.set_xlabel("x")
ax.set_ylabel("y")
cb3=plt.colorbar(im3, ax=ax)
cb3.set_label("conductivity (normalized)")
fig.savefig("conductivity.png",dpi=1200)
plt.close(fig)
