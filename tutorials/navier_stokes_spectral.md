# Neural closure models for the incompressible Navier-Stokes equations

## The incompressible Navier-Stokes equations

The incompressible Navier-Stokes equations in a periodic box $\Omega = [0,
1]^2$ are comprised of the mass equation

$$
\nabla \cdot u = 0
$$

and the momentum equation

$$
\frac{\partial u}{\partial t} = - \nabla p - \nabla \cdot (u u^\mathsf{T}) +
\nu \nabla^2 u + f,
$$

where $p$ is the pressure, $u = (u_x, u_y)$ is the velocity, and $f = (f_x,
f_y)$ is the body force.

We can represent the solution in spectral space as follows:

$$
u(x, t) = \sum_{k \in \mathbb{Z}^2} \hat{u}(k, t) \mathrm{e}^{\mathrm{i}
k^\mathsf{T} x}.
$$

Instead of the continuous solution $u$, we now have a countable number of
coefficients $\hat{u}$.

The mass equation then takes the form

$$
\mathrm{i} k^\mathsf{T} \hat{u} = 0
$$

and similarly for the momentum equations:

$$
\begin{split}
\frac{\partial \hat{u}}{\partial t} & = - \mathrm{i} k \hat{p} - \mathrm{i}
\widehat{u u^\mathsf{T}} k - \nu \| k \|^2 \hat{u} + \hat{f} \\
& =  - \mathrm{i} k \hat{p} + \hat{F}(\hat{u}),
\end{split}
$$

where $k = (k_x, k_y)$ is the wave number, $\hat{p}$ is the Fourier
coefficients of $p$, and similarly for $\hat{u} = (\hat{u}_x, \hat{u}_y)$ and
$\hat{f} = (\hat{f}_x, \hat{f}_y)$. These equations are obtained by replacing
$\nabla$ with $\mathrm{i} k$. We will also name the nonlinear (quadratic)
term $Q = - \mathrm{i} \widehat{u u^\mathsf{T}} k$. Note that the non-linear
term $u u^\mathsf{T}$ is still computed in physical space, as computing it in
spectral space would require evaluating a convolution integral instead of a
point-wise product. Note also that since the domain $\Omega$ is compact, the
modes are countable, meaning that $k \in \mathbb{Z}^2$, paving the way for
discretization by truncation.

Taking the time derivative of the mass equation gives a spectral Poisson
equation for the pressure:

$$
- \| k \|^2 \hat{p} = i k^\mathsf{T} \hat{F}(\hat{u})
= - k^\mathsf{T} \widehat{u u^\mathsf{T}} k + \mathrm{i} k^\mathsf{T} \hat{f}.
$$

The pressure solution is however not defined for $k = 0$, as the pressure is only
determined up to a constant in the case of a periodic domain.
By setting $\hat{p}(k = 0) = 0$, we fix this
constant at zero.

In the following, we will make use of a pressure-free equation (for $k \neq 0$):

$$
\frac{\partial \hat{u}}{\partial t} = \left(1 - \frac{k
k^\mathsf{T}}{\| k \|^2} \right) \hat{F}(\hat{u}) = P \hat{F}(\hat{u}),
$$

where the pressure gradient is replaced by the mode-wise projection
operator $P(k) \in \mathbb{C}^{2 \times 2}$. Note
that we set $P(k = 0) = I$, thus preserving averages.

## Spatial discretization

A natural way to discretize the pseudo-spectral Navier-Stokes equations is to
truncate at a maximum frequency $K$. However, the non-linear term may create
wave-numbers up to $2 K$ when the input $u$ has wave numbers up to $K$.
These additional wavenumbers are aliased with the lower resolved ones. We
will therefore compute the non-linear term from the $2 / 3$ lowest
wave-numbers of $u$ only (a common heuristic [^1]).

The spatial solution $u(x, y, t)$ will be represented on a uniform grid
$x = y = (i / n)_{i = 1}^N$, where $N = 2 K$. Using the convention of the fast Fourier
transform (FFT) [^2], we index the spectral fields by a vector of wave numbers $k_x = k_y = (0, 1, \dots, K -
1, -K, -(K - 1), \dots 1) \in \mathbb{Z}^N$.

````julia
using ComponentArrays
using CUDA
using FFTW
using IJulia
using LinearAlgebra
using Lux
using LuxCUDA
using NNlib
using Optimisers
using Plots
using Printf
using Random
using Zygote
````

Lux likes to toss random number generators around, for reproducible science

````julia
rng = Random.default_rng()
````

We define some useful functions, starting with `zeros`.

````julia
z = CUDA.functional() ? CUDA.zeros : (s...) -> zeros(Float32, s...)
ArrayType = CUDA.functional() ? CuArray : Array
````

This line makes sure that we don't do accidental CPU stuff while things
should be on the GPU

````julia
CUDA.allowscalar(false)
````

Since most of the manipulations take place in spectral space, we drop the
hats, e.g. `u` is $\hat{u}$. Also, `u` will have the shape `(N, N, 2)`.

The function `Q` computes the quadratic term.
The `K - Kf` highest frequencies of `u` are cut-off to prevent aliasing.

````julia
function Q(u, params)
    (; K, Kf, k) = params
    n = size(u, 1)
    Kz = K - Kf

    # Remove aliasing components
    uf = [
        u[1:Kf, 1:Kf, :] z(Kf, 2Kz, 2) u[1:Kf, end-Kf+1:end, :]
        z(2Kz, Kf, 2) z(2Kz, 2Kz, 2) z(2Kz, Kf, 2)
        u[end-Kf+1:end, 1:Kf, :] z(Kf, 2Kz, 2) u[end-Kf+1:end, end-Kf+1:end, :]
    ]

    # Spatial velocity
    v = real.(ifft(uf, (1, 2)))
    vx, vy = eachslice(v; dims = 3)

    # Quadractic terms in space
    vxx = vx .* vx
    vxy = vx .* vy
    vyy = vy .* vy
    v2 = cat(vxx, vxy, vxy, vyy; dims = 3)
    v2 = reshape(v2, n, n, 2, 2)

    # Quadractic terms in spectral space
    q = fft(v2, (1, 2))
    qx, qy = eachslice(q; dims = 4)

    # Compute partial derivatives in spectral space
    ikx = im * k
    iky = im * reshape(k, 1, :)
    q = @. -ikx * qx - iky * qy

    # Zero out high wave-numbers (is this necessary?)
    q = [
        q[1:Kf, 1:Kf, :] z(Kf, 2Kz, 2) q[1:Kf, Kf+2Kz+1:end, :]
        z(2Kz, Kf, 2) z(2Kz, 2Kz, 2) z(2Kz, Kf, 2)
        q[Kf+2Kz+1:end, 1:Kf, :] z(Kf, 2Kz, 2) q[Kf+2Kz+1:end, Kf+2Kz+1:end, :]
    ]

    q
end
````

`F` computes the unprojected momentum right hand side $\hat{F}$. It also
includes the closure term (if any).

````julia
function F(u, params)
    (; normk, nu, f, m, θ) = params
    q = Q(u, params)
    du = @. q - nu * normk * u + f
    isnothing(m) || (du += m(u, θ))
    du
end
````

The projector $P$ uses pre-assembled matrices.

````julia
function project(u, params)
    (; Pxx, Pxy, Pyy) = params
    ux, uy = eachslice(u; dims = 3)
    dux = @. Pxx * ux + Pxy * uy
    duy = @. Pxy * ux + Pyy * uy
    cat(dux, duy; dims = 3)
end
````

## Time discretization

For time stepping, we do a simple fourth order explicit Runge-Kutta scheme.

From a current state $\hat{u}^0 = \hat{u}(t)$, we divide the outer time step
$\Delta t$ into $s = 4$ sub-steps as follows:

$$
\begin{split}
\hat{F}^i & = P \hat{F}(\hat{u}^{i - 1}) \\
\hat{u}^i & = u^0 + \Delta t \sum_{j = 1}^{i} a_{i j} F^j.
\end{split}
$$

The solution at the next outer time step $t + \Delta t$ is then
$\hat{u}^s = \hat{u}(t + \Delta t) + \mathcal{O}(\Delta t^5)$. The coefficients
of the RK method are chosen as

$$
a = \begin{pmatrix}
    \frac{1}{2} & 0           & 0           & 0 \\
    0           & \frac{1}{2} & 0           & 0 \\
    0           & 0           & 1           & 0 \\
    \frac{1}{6} & \frac{2}{6} & \frac{2}{6} & \frac{1}{6}
\end{pmatrix}.
$$

Note that each of the intermediate steps is divergence free.

The following function performs one RK4 time step. Note that we never
modify any vectors, only create new ones. The AD-framework Zygote prefers
it this way.

````julia
function step_rk4(u0, params, dt)
    a = (
        (0.5f0,),
        (0.0f0, 0.5f0),
        (0.0f0, 0.0f0, 1.0f0),
        (1.0f0 / 6.0f0, 2.0f0 / 6.0f0, 2.0f0 / 6.0f0, 1.0f0 / 6.0f0),
    )
    u = u0
    k = ()
    for i = 1:length(a)
        ki = project(F(u, params), params)
        k = (k..., ki)
        u = u0
        for j = 1:i
            u += dt * a[i][j] * k[j]
        end
    end
    u
end
````

For plotting, the spatial vorticity can be useful. It is given by

$$
\omega = -\frac{\partial u_x}{\partial y} + \frac{\partial u_y}{\partial x},
$$

which becomes

$$
\hat{\omega} = - \mathrm{i} k_y u_x + \mathrm{i} k_x u_y
$$

in spectral space.

````julia
function vorticity(u, params)
    (; k) = params
    ikx = im * k
    iky = im * reshape(k, 1, :)
    ux, uy = eachslice(u; dims = 3)
    ω = @. -iky * ux + ikx * uy
    real.(ifft(ω))
end
````

This function creates a random Gaussian force field.

````julia
function gaussian(x; σ = 0.1f0)
    n = length(x)
    xf, yf = rand(), rand()
    f = [
        exp(-(x - xf + a)^2 / σ^2 - (y - yf + a)^2 / σ^2) for x ∈ x, y in x,
        a in (-1, 0, 1), b in (-1, 0, 1)
    ] ## periodic padding
    f = reshape(sum(f; dims = (3, 4)), n, n)
    f = exp(im * rand() * 2.0f0π) * f ## Rotate f
    cat(real(f), imag(f); dims = 3)
end
````

For the initial conditions, we create a random spectrum with some decay.
Note that the initial conditions are projected onto the divergence free
space at the end.

````julia
function create_spectrum(params; A, σ, s)
    (; x, k, K) = params
    T = eltype(x)
    kx = k
    ky = reshape(k, 1, :)
    a = z(2K, 2K)
    a = a .+ (1 + 0im)
    τ = 2.0f0π
    a = @. A / sqrt(τ^2 * 2σ^2) *
       exp(-(kx - s)^2 / 2σ^2 - (ky - s)^2 / 2σ^2 - im * τ * rand(T))
    a
end

function random_field(params; A = 1.0f6, σ = 30.0f0, s = 5.0f0)
    ux = create_spectrum(params; A, σ, s)
    uy = create_spectrum(params; A, σ, s)
    u = cat(ux, uy; dims = 3)
    u = real.(ifft(u, (1, 2)))
    u = fft(u, (1, 2))
    project(u, params)
end
````

Body force

````julia
# f = 10 * (10 * gaussian(x) + 15 * gaussian(x) + 3 * gaussian(x))
# heatmap(selectdim(f, 3, 1))
# heatmap(selectdim(f, 3, 2))
# f = fft(f, (1, 2))
# heatmap(abs.(selectdim(f, 3, 1)))
# heatmap(abs.(selectdim(f, 3, 2)))

# y = x'
# fx = @. 100.0f0 * sin(8.0f0π * y) + 0 * x
# fy = @. 2 * x * y * sin(16.0f0π * y)
# f = cat(fx, fy; dims = 3)
# f = fft(f, (1, 2))
#
# heatmap(fx)
# heatmap(fy)
````

Store paramaters and precomputed operators in a named tuple to toss around.
Having this in a function gets useful when we later work with multiple
resolutions.

````julia
function create_params(
    K;
    nu,
    f = z(2K, 2K),
    m = nothing,
    θ = nothing,
    anti_alias_factor = 2 / 3,
)
    Kf = round(Int, anti_alias_factor * K)
    N = 2K
    x = LinRange(0.0f0, 1.0f0, N + 1)[2:end]

    # Vector of wave numbers

    k = ArrayType(fftfreq(N, N))
    normk = k .^ 2 .+ k' .^ 2

    # Projection components
    kx = k
    ky = reshape(k, 1, :)
    Pxx = @. 1 - kx * kx / (kx^2 + ky^2)
    Pxy = @. 0 - kx * ky / (kx^2 + ky^2)
    Pyy = @. 1 - ky * ky / (kx^2 + ky^2)

    # The zero'th component is currently `0/0 = NaN`. For `CuArray`s,
    # we need to explicitly allow scalar indexing.

    CUDA.@allowscalar Pxx[1, 1] = 1
    CUDA.@allowscalar Pxy[1, 1] = 0
    CUDA.@allowscalar Pyy[1, 1] = 1

    # Closure model
    m = nothing
    θ = nothing

    (; x, N, K, Kf, k, nu, normk, f, Pxx, Pxy, Pyy, m, θ)
end
````

## Example simulation

Let's test our method in action.

````julia
params = create_params(64; nu = 0.001f0)

# Initial conditions
u = random_field(params)
````

We can also check that `u` is indeed divergence free

````julia
maximum(abs, params.k .* u[:, :, 1] .+ params.k' .* u[:, :, 2])
````

Let's do some time stepping.

````julia
t = 0.0f0
dt = 1.0f-3

for i = 1:1000
    t += dt
    u = step_rk4(u, params, dt)
    if i % 10 == 0
        ω = Array(vorticity(u, params))
        title = @sprintf("Vorticity, t = %.3f", t)
        fig = heatmap(ω'; xlabel = "x", ylabel = "y", title)
        display(fig)
        sleep(0.001) # Time for plot
    end
end
````

Well, that looks like... a fluid! In 2D, the eddies will eventually just
fade and merge in the absence of forcing. We could of course add a force
also.

## Filtering and large eddy simulation (LES)

We now assume that a resolution $K_\text{DNS}$ is sufficient to resolve the
smallest structures of the flow. The resulting solution will be denoted
$\hat{u}(k, t)$, resulting from _direct numerical simulation_ (DNS). Since
this resolution is intractable, we will instead do _Large Eddy Simulation_
(LES), at a much coarser resolution. The goal of our LES simulation is that
the obtained solution $\bar{\hat{v}}$ is similar to the "filtered DNS"
solution $\bar{\hat{u}}(k) = \phi(k) \hat{u}(k)$. We here define it using a
spectral cut-off filter, where $\bar{\hat{u}}(k) = \hat{u}(k)$ for $k \in
\{-K_\text{LES}, \dots, K_\text{LES} - 1 \}$ with $K_\text{LES} <
K_\text{DNS}$.

The filtered solution $\bar{\hat{u}}$ is governed by the equations

$$
\frac{\mathrm{d} \bar{\hat{u}}}{\mathrm{d} t} = \bar{P} \left(
\bar{\hat{F}}(\bar{\hat{u}}) + c(\hat{u}, \bar{\hat{u}}) \right),
$$

where $\bar{P}$ and $\bar{\hat{F}}$ are the coarse-resolution version of $P$
and $\hat{F}$, and $c = \overline{\hat{Q}(\hat{u})} -
\bar{\hat{Q}}(\bar{\hat{u}})$ is the commutator error (only present in the
quadratic term for spectral filters). This commutator error is predicted
using a closure model $m$ with parameters $\theta$. The resulting closed
system produces a predicted velocity $\bar{\hat{v}}$:

$$
\frac{\mathrm{d} \bar{\hat{v}}}{\mathrm{d} t} = \bar{P} \left(
\bar{\hat{F}}(\bar{\hat{v}}) + m(\bar{\hat{v}}, \theta) \right).
$$

### Model architecture

We are free to choose the model architecture $m$.

#### Fourier neural operator architecture

Now let's implement the Fourier Neural Operator (FNO) [^3].
A Fourier layer $u \mapsto w$ is given by the following expression:

$$
w(x) = \sigma \left( z(x) + W u(x) \right)
$$

where $\hat{z}(k) = R(k) \hat{u}(k)$ for some weight matrix collection
$R(k) \in \mathbb{C}^{n_\text{out} \times n_\text{in}}$. The important
part is the following choice: $R(k) = 0$ for $\| k \| > k_\text{max}$
for some $k_\text{max}$. This truncation makes the FNO applicable to
any discretization as long as $K > k_\text{max}$, and the same parameters
may be reused.

The deep learning framework [Lux](https://lux.csail.mit.edu/) let's us define
our own layer types. Everything should be explicit ("functional
programming"), including random number generation and state modification. The
weights are stored in a vector outside the layer, while the layer itself
contains information for construction the network.

````julia
struct FourierLayer{A,F} <: Lux.AbstractExplicitLayer
    kmax::Int
    cin::Int
    cout::Int
    σ::A
    init_weight::F
end

FourierLayer(kmax, ch::Pair{Int,Int}; σ = identity, init_weight = glorot_uniform) =
    FourierLayer(kmax, first(ch), last(ch), σ, init_weight)
````

We also need to specify how to initialize the parameters and states. The
Fourier layer does not have any hidden states (RNGs) that are modified.

````julia
Lux.initialparameters(rng::AbstractRNG, (; kmax, cin, cout, init_weight)::FourierLayer) = (;
    spatial_weight = init_weight(rng, cout, cin),
    spectral_weights = init_weight(rng, kmax + 1, kmax + 1, cout, cin, 2),
)
Lux.initialstates(::AbstractRNG, ::FourierLayer) = (;)
Lux.parameterlength((; kmax, cin, cout)::FourierLayer) =
    cout * cin + (kmax + 1)^2 * 2 * cout * cin
Lux.statelength(::FourierLayer) = 0
````

We now define how to pass inputs through Fourier layer, assuming the
following:

- Input size: `(N, N, cin, nsample)`
- Output size: `(N, N, cout, nsample)`

````julia
function ((; kmax, cout, cin, σ)::FourierLayer)(x, params, state)
    N = size(x, 1)

    # Destructure params
    # The real and imaginary parts of R are stored in two separate channels
    W = params.spatial_weight
    W = reshape(W, 1, 1, cout, cin)
    R = params.spectral_weights
    R = selectdim(R, 5, 1) .+ im .* selectdim(R, 5, 2)

    # Spatial part (applied point-wise)

    y = reshape(x, N, N, 1, cin, :)
    y = sum(W .* y; dims = 4)
    y = reshape(y, N, N, cout, :)

    # Spectral part (applied mode-wise)
    #
    # Steps:
    #
    # - go to complex-valued spectral space
    # - chop off high wavenumbers
    # - multiply with weights mode-wise
    # - pad with zeros to restore original shape
    # - go back to real valued spatial representation
    ikeep = (1:kmax+1, 1:kmax+1)
    nkeep = (kmax + 1, kmax + 1)
    dims = (1, 2)
    z = fft(x, dims)
    z = z[ikeep..., :, :]
    z = reshape(z, nkeep..., 1, cin, :)
    z = sum(R .* z; dims = 4)
    z = reshape(z, nkeep..., cout, :)
    z = pad_zeros(z, (0, N - kmax - 1, 0, N - kmax - 1); dims)
    z = real.(ifft(z, dims))

    # Outer layer: Activation over combined spatial and spectral parts
    # Note: Even though high wavenumbers are chopped off in `z` and may
    # possibly not be present in the input at all, `σ` creates new high
    # wavenumbers. High wavenumber functions may thus be represented using a
    # sequence of Fourier layers. In this case, the `y`s are the only place
    # where information contained in high input wavenumbers survive in a
    # Fourier layer.
    v = σ.(y .+ z)

    # Fourier layer does not modify state
    v, state
end
````

We will use four Fourier layers, with a final dense layer.
Since the closure is applied in spectral space, we start and end there.

````julia
# Number of channels
ch_fno = [2, 5, 5, 5, 2]

# Cut-off wavenumbers
kmax_fno = [8, 8, 8, 8]

# Fourier layer activations
σ_fno = [gelu, gelu, gelu, identity]

# Model
_fno = Chain(
    # Go to physical space
    u -> real.(ifft(u, (1, 2))),

    # Some Fourier layers
    (
        FourierLayer(kmax_fno[i], ch_fno[i] => ch_fno[i+1]; σ = σ_fno[i]) for
        i ∈ eachindex(σ_fno)
    )...,

    # Put channels in first dimension
    u -> permutedims(u, (3, 1, 2, 4)),

    # Compress with a final dense layer
    Dense(ch_fno[end] => 2 * ch_fno[end], gelu),
    Dense(2 * ch_fno[end] => 2; use_bias = false),

    # Put channels back after spatial dimensions
    u -> permutedims(u, (2, 3, 1, 4)),

    # Go to spectral space
    u -> fft(u, (1, 2)),
)
````

Create parameter vector and empty state

````julia
θ_fno, state_fno = Lux.setup(rng, _fno)
θ_fno = gpu_device()(ComponentArray(θ_fno))
length(θ_fno)
````

````julia
fno(v, θ) = first(_fno(v, θ, state_fno))
````

#### Convolutional neural network

Alternatively, we may use a CNN closure model. There should be fewer
parameters.

````julia
# Radius
r_cnn = [2, 2, 2, 2]

# Channels
ch_cnn = [2, 8, 8, 8, 2]

# Activations
σ_cnn = [leakyrelu, leakyrelu, leakyrelu, identity]

# Bias
b_cnn = [true, true, true, false]

_cnn = Chain(
    # Go to physical space
    u -> real.(ifft(u, (1, 2))),

    # Add padding so that output has same shape as commutator error
    u -> pad_circular(u, sum(r_cnn)),

    # Some convolutional layers
    (
        Conv(
            (2 * r_cnn[i] + 1, 2 * r_cnn[i] + 1),
            ch_cnn[i] => ch_cnn[i+1],
            σ_cnn[i];
            use_bias = b_cnn[i],
        ) for i ∈ eachindex(r_cnn)
    )...,

    # Go to spectral space
    u -> fft(u, (1, 2)),
)
````

Create parameter vector and empty state

````julia
θ_cnn, state_cnn = Lux.setup(rng, _cnn)
θ_cnn = gpu_device()(ComponentArray(θ_cnn))
length(θ_cnn)
````

````julia
cnn(v, θ) = first(_cnn(v, θ, state_cnn))
````

### Choosing model parameters: loss function

Ideally, we want LES to produce the filtered DNS velocity $\bar{\hat{u}}$. We
can thus minimize

$$
L(\theta) = \| \bar{\hat{v}}_\theta - \bar{\hat{u}} \|^2.
$$

Alternatively, we can minimize the simpler loss function

$$
L(\theta) = \| m(\bar{\hat{u}}, \theta) - c(\hat{u}, \bar{\hat{u}}) \|^2.
$$

This data-driven minimization will give us $\theta$.

Random a priori loss function for stochastic gradient descent

````julia
mean_squared_error(f, x, y, θ; λ = 1.0f-4) =
    sum(abs2, f(x, θ) - y) / sum(abs2, y) + λ * sum(abs2, θ) / length(θ)

function create_randloss(f, x, y; nuse = size(x, 2))
    d = ndims(x)
    nsample = size(x, d)
    function randloss(θ)
        i = Zygote.@ignore sort(shuffle(1:nsample)[1:nuse])
        xuse = Zygote.@ignore ArrayType(selectdim(x, d, i))
        yuse = Zygote.@ignore ArrayType(selectdim(y, d, i))
        mean_squared_error(f, xuse, yuse, θ)
    end
end
````

Random trajectory (a posteriori) loss function

````julia
function trajectory_loss(ubar, θ; params, dt = 1.0f-3)
    nt = size(ubar, 4)
    loss = 0.0f0
    v = ubar[:, :, :, 1]
    for i = 2:nt
        v = step_rk4(v, (; params..., θ), dt)
        u = ubar[:, :, :, i]
        loss += sum(abs2, v - u) / sum(abs2, u)
    end
    loss
end

function create_randloss_trajectory(ubar; params, dt, nunroll = 10)
    d = ndims(ubar)
    nt = size(ubar, d)
    function randloss(θ)
        istart = Zygote.@ignore rand(1:nt-nunroll)
        trajectory = Zygote.@ignore ArrayType(selectdim(ubar, d, istart:istart+nunroll))
        trajectory_loss(trajectory, θ; params, dt)
    end
end
````

### Data generation

Create some filtered DNS data (one initial condition only)

````julia
nu = 0.001f0
params_les = create_params(32; nu)
params_dns = create_params(128; nu)

# Initial conditions
u = random_field(params_dns)

# Let's do some time stepping.

t = 0.0f0
dt = 1.0f-3
nt = 1000

# Filtered snapshots
v = zeros(Complex{Float32}, params_les.N, params_les.N, 2, nt + 1)

# Commutator errors
c = zeros(Complex{Float32}, params_les.N, params_les.N, 2, nt + 1)

spectral_cutoff(u, K) = [
    u[1:K, 1:K, :] u[1:K, end-K+1:end, :]
    u[end-K+1:end, 1:K, :] u[end-K+1:end, end-K+1:end, :]
]

for i = 1:nt+1
    if i > 1
        t += dt
        u = step_rk4(u, params_dns, dt)
    end
    ubar = spectral_cutoff(u, params_les.K)
    v[:, :, :, i] = Array(ubar)
    c[:, :, :, i] =
        Array(spectral_cutoff(F(u, params_dns), params_les.K) - F(ubar, params_les))
    if i % 10 == 0
        # println(i)
        ω = Array(vorticity(u, params_dns))
        title = @sprintf("Vorticity, t = %.3f", t)
        fig = heatmap(ω'; xlabel = "x", ylabel = "y", title)
        display(fig)
        sleep(0.001) # Time for plot
    end
end
````

Choose closure model

````julia
m, θ₀ = fno, θ_fno
# m, θ₀ = cnn, θ_cnn
````

Choose loss function

````julia
randloss = create_randloss(m, v, c; nuse = 50)
# randloss = create_randloss_trajectory(v; params = (; params_les..., m), dt = 1f-3, nunroll = 10)
````

Model warm-up: trigger compilation and get indication of complexity

````julia
randloss(θ₀)
gradient(randloss, θ₀);
@time randloss(θ₀);
@time gradient(randloss, θ₀);
````

### Training

We will monitor the error along the way.

````julia
θ = θ₀
v_test, c_test = ArrayType(v[:, :, :, end:end]), ArrayType(c[:, :, :, end:end])
opt = Optimisers.setup(Adam(1.0f-3), θ)
ncallback = 1
ntrain = 100
ihist = Int[]
ehist = Float32[]
ishift = 0
````

The cell below can be repeated

````julia
for i = 1:ntrain
    g = first(gradient(randloss, θ))
    opt, θ = Optimisers.update(opt, θ, g)
    if i % ncallback == 0
        e = norm(m(v_test, θ) - c_test) / norm(c_test)
        push!(ihist, ishift + i)
        push!(ehist, e)
        fig = plot(; xlabel = "Iterations", title = "Relative a-priori error")
        hline!(fig, [1.0f0]; linestyle = :dash, label = "No model")
        plot!(fig, ihist, ehist; label = "FNO")
        display(fig)
    end
end
ishift += ntrain
````

-

````julia
GC.gc()
CUDA.reclaim()
````

# See also

- <https://github.com/FourierFlows/FourierFlows.jl>

[^1]: S. A. Orszag. _On the elimination of aliasing in finite-difference
      schemes by filtering high-wavenumber components._ Journal of the
      Atmospheric Sciences 28, 1074-107 (1971).
[^2]: <https://en.wikipedia.org/wiki/Fast_Fourier_transform>
[^3]: Z. Li, N. Kovachki, K. Azizzadenesheli, B. Liu, K. Bhattacharya, A. Stuart, and
      A. Anandkumar.  _Fourier neural operator for parametric partial differential
      equations._ arXiv:[2010.08895](https://arxiv.org/abs/2010.08895), 2021.

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

