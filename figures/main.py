import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['text.usetex']=False
plt.rcParams['svg.fonttype']='none'
plt.rcParams['font.size']=8

# Grid parameters
iters = 8000
tol = 1e-6

sig_BULK = 1.0
sig_FLAW = 1e-3

# find harmonic mean
def h_mean(a, b, eps=1e-12):
    return (2.0 * a * b) / (a + b + eps)

# find electric field
def e_field(V, hx=1.0, hy=1.0):
    dVdi, dVdj = np.gradient(V, hx, hy, edge_order=2)
    Ex = -dVdj
    Ey = -dVdi
    return Ex, Ey

# solve for conductance s and voltage V over the whole field using SOR
def conductance(Nx=60, Ny=60, isFlawed=False, omega=1.8):

    s = np.ones((Nx, Ny)) * sig_BULK

    # scale flaw to grid size because it's not in same place otherwise
    if isFlawed:
        i_start = max(1, int(0.17 * Nx))
        i_end   = max(i_start+1, int(0.83 * Nx))

        j_start = max(1, int(0.50 * Ny))
        j_end   = max(j_start+1, int(0.53 * Ny))

        s[i_start:i_end, j_start:j_end] = sig_FLAW

    V = np.zeros((Nx, Ny))
    for j in range(Ny):
        V[:, j] = 1.0 - (j / (Ny - 1))

    fixed = np.zeros((Nx, Ny), dtype=bool)
    fixed[:, 0] = True
    fixed[:, -1] = True

    # apply boundary conditions to the array
    def apply_bc(V):
        V[:, 0] = 1.0
        V[:, -1] = 0.0
        V[0, :] = V[1, :]
        V[-1, :] = V[-2, :]

    apply_bc(V)

    for iteration in range(iters):

        max_d = 0.0

        for i in range(1, Nx - 1):
            for j in range(1, Ny - 1):

                # if the voltage is fixed already move on
                # nb this is not quite an insulating bc it is a pinned bc
                if fixed[i, j]:
                    continue

                ip, im = i + 1, i - 1
                jp, jm = j + 1, j - 1

                sxp = h_mean(s[i, j], s[ip, j])
                sxm = h_mean(s[i, j], s[im, j])
                syp = h_mean(s[i, j], s[i, jp])
                sym = h_mean(s[i, j], s[i, jm])

                A = sxp + sxm + syp + sym
                # if the conductance of the neighbors are small skip to optimize
                if A < 1e-20:
                    continue

                # Gauss-Seidel here
                V_next = (
                    sxp * V[ip, j] + sxm * V[im, j] +
                    syp * V[i, jp] + sym * V[i, jm]
                ) / A

                # Successive Over Relaxation here
                V_prev = V[i, j]
                V[i, j] = (1.0 - omega) * V_prev + omega * V_next

                d = abs(V[i, j] - V_prev)
                if d > max_d:
                    max_d = d

        apply_bc(V)

        # solution has converged
        if max_d < tol:
            break

    return V, s

# arrows for direction of the potential field
def add_arrowheads(ax, stream, color="white", lw=1.0):
    segs = stream.lines.get_segments()

    if len(segs) == 0:
        return

    step = 12
    for idx in range(0, len(segs), step):
        seg = segs[idx]
        (x0, y0), (x1, y1) = seg[0], seg[1]

        ax.annotate(
            r"",
            xy=(x1, y1),
            xytext=(x0, y0),
            arrowprops=dict(
                arrowstyle="->",
                color=color,
                linewidth=lw,
                shrinkA=0,
                shrinkB=0,
            ),
        )


# test with different gird resolutions and plot each
for N in [30, 60, 120]:

    V_clean, sig_clean = conductance(Nx=N, Ny=N, isFlawed=False)
    V_flaw, sig_flaw = conductance(Nx=N, Ny=N, isFlawed=True)

    V_diff = V_flaw - V_clean

    Ex_clean, Ey_clean = e_field(V_clean)
    Ex_flaw, Ey_flaw = e_field(V_flaw)

    #fig, axs = plt.subplots(1, 4, figsize=(13.6668, 2), dpi=600, constrained_layout=True)
    


    x = np.arange(N)
    y = np.arange(N)
    X, Y = np.meshgrid(x, y)

    # potential plot with no flaw
    fig,ax = plt.subplots(figsize=(3.4167,3),dpi=1200,layout="constrained")
    im0 = ax.imshow(V_clean, origin="lower", cmap="viridis", vmin=0, vmax=1,
                    extent=[0, N-1, 0, N-1],rasterized=False)
    s0 = ax.streamplot(X, Y, Ex_clean, Ey_clean,
                       density=0.6, linewidth=0.9,
                       color=(1,1,1,0.75), arrowstyle='-')
    add_arrowheads(ax, s0, color=(1,1,1,0.9), lw=1.0)

    #axs[0].set_title(f"Clean (N={N})")
    ax.set_xlabel(r'$x$, nodes',parse_math=False)
    ax.set_ylabel(r'$y$, nodes',parse_math=False)

    cb0 = plt.colorbar(im0, ax=ax)
    cb0.set_label(r'$V$, \unit{\volt}',parse_math=False)

    #fig.tight_layout()
    fig.savefig('potential-noflaw-{0}.svg'.format(N))
    fig.savefig('potential-noflaw-{0}.png'.format(N),dpi=1200)
    plt.close(fig)

    

    # potential plot with flaw
    fig,ax = plt.subplots(figsize=(3.4167,3),dpi=1200,layout="constrained")
    im1 = ax.imshow(V_flaw, origin="lower", cmap="viridis", vmin=0, vmax=1,
                    extent=[0, N-1, 0, N-1],rasterized=False)
    s1 = ax.streamplot(X, Y, Ex_flaw, Ey_flaw,
                       density=0.6, linewidth=0.9,
                       color=(1,1,1,0.75), arrowstyle='-')
    add_arrowheads(ax, s1, color=(1,1,1,0.9), lw=1.0)

    #axs[1].set_title(f"Flaw (N={N})")
    ax.set_xlabel(r'$x$, nodes',parse_math=False)
    ax.set_ylabel(r'$y$, nodes',parse_math=False)

    cb1 = plt.colorbar(im1, ax=ax)
    cb1.set_label(r'$V$, \unit{\volt}',parse_math=False)

    #fig.tight_layout()
    fig.savefig('potential-flaw-{0}.svg'.format(N))
    fig.savefig('potential-flaw-{0}.png'.format(N),dpi=1200)
    plt.close(fig)



    
    
    # voltage differnce plot
    fig,ax = plt.subplots(figsize=(3.4167,3),dpi=1200,layout="constrained")
    dvmax = np.max(np.abs(V_diff))

    im2 = ax.imshow(V_diff, origin="lower", cmap="seismic",
                    vmin=-dvmax, vmax=dvmax,
                    extent=[0, N-1, 0, N-1],rasterized=False)

    #axs[2].set_title(f"ΔV (N={N})")
    ax.set_xlabel(r'$x$, nodes',parse_math=False)
    ax.set_ylabel(r'$y$, nodes',parse_math=False)

    cb2 = plt.colorbar(im2, ax=ax)
    cb2.set_label(r'$\Delta V$, \unit{\volt}',parse_math=False)

    #fig.tight_layout()
    fig.savefig('potential-difference-{0}.svg'.format(N))
    fig.savefig('potential-difference-{0}.png'.format(N),dpi=1200)
    plt.close(fig)



    
    # conductivity plot
    fig,ax = plt.subplots(figsize=(3.4167,3),dpi=1200,layout="constrained")
    im3 = ax.imshow(sig_flaw, origin="lower", cmap="gray_r",
                    extent=[0, N-1, 0, N-1],rasterized=False)

    #axs[3].set_title(f"Conductivity (N={N})")
    ax.set_xlabel(r'$x$, nodes',parse_math=False)
    ax.set_ylabel(r'$y$, nodes',parse_math=False)

    cb3 = plt.colorbar(im3, ax=ax)
    cb3.set_label(r'conductivity $\sigma$, normalized',parse_math=False)

    #fig.tight_layout()
    fig.savefig('conductivity-{0}.svg'.format(N))
    fig.savefig('conductivity-{0}.png'.format(N),dpi=1200)
    plt.close(fig)

    #plt.savefig(f"interior_fieldsig_{N}.svg")
    #plt.savefig(f"interior_fieldsig_{N}.png", dpi=600)
    #plt.show()
