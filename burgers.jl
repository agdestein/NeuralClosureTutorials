# # Neural closure models for the viscous Burgers equation

#-

# ## Running on Google Colab
#
# _This section is only needed when running on Google colab._
#
# To use Julia on Google colab, we will install Julia using the official version
# manager Juliup. From the default Python kernel, we can access the shell by
# starting a line with `!`.

#nb !curl -fsSL https://install.julialang.org | sh -s -- --yes

# We can check that Julia is successfully installed on the Colab instance.

#nb !/root/.juliaup/bin/julia -e 'println("Hello")'

# We now proceed to install the necessary Julia packages, including `IJulia` which
# will add the Julia notebook kernel.

#nb %%shell
#nb /root/.juliaup/bin/julia -e '''
#nb     using Pkg
#nb     Pkg.add([
#nb         "ComponentArrays",
#nb         "FFTW",
#nb         "IJulia",
#nb         "LinearAlgebra",
#nb         "Lux",
#nb         "LuxCUDA",
#nb         "NNlib",
#nb         "Optimisers",
#nb         "Plots",
#nb         "Printf",
#nb         "Random",
#nb         "SparseArrays",
#nb         "Zygote",
#nb     ])
#nb '''

# Once this is done, do the following:
# 1. Reload the browser page (`CTRL`/`CMD` + `R`)
# 2. In the top right corner of Colab, then select the Julia kernel.

#-

# ## The viscous Burgers equation
#
# Consider the periodic domain $\Omega = [0, 1]$. The visous Burgers equation
# is given by
#
# $$
# \frac{\partial u}{\partial t} = - \frac{1}{2} \frac{\partial }{\partial x}
# \left( u^2 \right) + \frac{\nu}{\pi} \frac{\partial^2 u}{\partial x^2},
# $$
#
# where $\nu > 0$ is the viscosity.
#
# ### Discretization
#
# Consider a uniform discretization $x = \left( \frac{n}{N} \right)_{n =
# 1}^N$, with the additional point $x_0 = 0$ overlapping with $x_N$. The step
# size is $\Delta = \frac{1}{N}$. Using
# central finite difference, we get the discrete equations
#
# $$
# \frac{\mathrm{d} u_n}{\mathrm{d} t} = - \frac{1}{2} \frac{u_{n + 1}^2 - u_{n -
# 1}^2}{2 \Delta} + \frac{\nu}{\pi} \frac{u_{n + 1} - 2 u_n + u_{n - 1}}{\Delta^2},
# $$
#
# with the convention $u_0 = u_N$ and $u_{N + 1} = u_1$ (periodic extension). The
# degrees of freedom are stored in the vector $u = (u_n)_{n = 1}^N$. In vector
# notation, we will write this as $\frac{\mathrm{d} u}{\mathrm{d} t} = f(u)$.
# Solving this equation for sufficiently small $\Delta$ (large $N$) will be
# referred to as _direct numerical simulation_ (DNS), and can be expensive.

#-

# ### Preparing the simulations
#
# Julia comes with built in array functionality. Additional functionality is
# provided in various packages, some of which are available in the Standard
# Library (LinearAlgebra, Printf, Random, SparseArrays). Others are available in
# the General Registry, and can be added using the built in package manager Pkg,
# e.g. `using Pkg; Pkg.add("Plots")`. If you ran the Colab setup section, the
# packages should already be added.

using ComponentArrays
using FFTW
using LinearAlgebra
using Lux
using LuxCUDA
using NNlib
using Optimisers
using Plots
using Printf
using Random
using SparseArrays
using Zygote

# Lux likes to toss random number generators (RNGs) around, for reproducible
# science. We therefore need to initialize an RNG. The seed makes sure the same
# sequence of pseudo-random numbers are generated at each time the session is
# restarted.

Random.seed!(123)
rng = Random.default_rng()

# Deep learning functions usually use single precision floating point numbers by
# default, as this is preferred on GPUs.  Julia itself, on the other hand, is
# slightly opinionated towards double precision floating point numbers, e.g.
# `3.14`, `1 / 6` and `2π` are all of type `Float64` (their single precision
# counterparts would be the slightly more verbose `3.14f0`, `1f0 / 6 ==
# Float32(1 / 6)` and `2f0π`). For simplicity, we will only use `Float64`. The
# only place we will encounter `Float32` in this file is in the default neural
# network weight initializers, so here is an alternative weight initializer
# using double precision.

glorot_uniform_64(rng::AbstractRNG, dims...) = glorot_uniform(rng, Float64, dims...)

# We start by defining the right hand side function for a vector `u`, making
# sure to account for the peridic boundaries. The macro `@.` makes sure that all
# following operations are performed element-wise.

function f(u; ν)
    N = size(u, 1)
    u₊ = circshift(u, -1)
    u₋ = circshift(u, 1)
    @. -N / 4 * (u₊^2 - u₋^2) + N^2 * ν / π * (u₊ - 2u + u₋)
end

# ### Time discretization
#
# For time stepping, we do a simple fourth order explicit Runge-Kutta scheme.
#
# From a current state $u^0 = u(t)$, we divide the outer time step
# $\Delta t$ into $s = 4$ sub-steps as follows:
#
# $$
# \begin{split}
# F^i & = f(u^{i - 1}) \\
# u^i & = u^0 + \Delta t \sum_{j = 1}^{i} a_{i j} F^j.
# \end{split}
# $$
#
# The solution at the next outer time step $t + \Delta t$ is then
# $u^s = u(t + \Delta t) + \mathcal{O}(\Delta t^4)$. The coefficients
# of the RK method are chosen as
#
# $$
# a = \begin{pmatrix}
#     1 & 0 & 0 & 0 \\
#     0 & 1 & 0 & 0 \\
#     0 & 0 & 1 & 0 \\
#     \frac{1}{6} & \frac{2}{6} & \frac{2}{6} & \frac{1}{6}
# \end{pmatrix}.
# $$
#
# The following function performs one RK4 time step. Note that we never
# modify any vectors, only create new ones. The AD-framework Zygote prefers
# it this way. The syntax `params...` lets us pass a variable number of keyword
# arguments to `f` (for the above `f` there is only one: `ν`).

function step_rk4(f, u₀, dt; params...)
    a = [
        1 0 0 0
        0 1 0 0
        0 0 1 0
        1/6 2/6 2/6 1/6
    ]
    u = u₀
    k = ()
    for i = 1:length(a)
        ki = f(u; params...)
        k = (k..., ki)
        u = u₀
        for j = 1:i
            u += dt * a[i, j] * k[j]
        end
    end
    u
end

# Chaining individual time steps allows us to solve the ODE. We here add the
# option to call a callback function after each time step. Note that the path
# from the final output `u` is obtained by passing the inputs `u₀` and
# parameters `params` through a finite amount of computational steps, each of
# which should have a chain rule defined and recognized in the Zygote AD
# framework.  The ODE solve should be differentiable, as long as `f` is.

function solve_ode(f, u₀, dt, nt; callback = (u, t, i) -> nothing, ncallback = 1, params...)
    t = 0.0
    u = u₀
    for i = 1:nt
        t += dt
        u = step_rk4(f, u, dt; params...)
        if i % ncallback == 0
            callback(u, t, i)
        end
    end
    u
end

# For the initial conditions, we create a random spectrum with some spectral
# amplitude decay profile.

function create_initial_conditions(x, nsample; kmax = 10, decay = k -> 1)
    # Fourier basis
    basis = [exp(2π * im * k * x) for x ∈ x, k ∈ -kmax:kmax]

    # Fourier coefficients with random phase and amplitude
    c = [randn() * exp(-2π * im * rand()) * decay(k) for k ∈ -kmax:kmax, _ ∈ 1:nsample]

    # Random data samples (real-valued)
    real.(basis * c)
end

# ### Example simulation
#
# Let's test our method in action.

N = 256
x = LinRange(0.0, 1.0, N + 1)[2:end]

## Initial conditions (one sample vector)
u = create_initial_conditions(x, 1; decay = k -> 1 / (1 + abs(k))^1.2)

## Let's do some time stepping
ν = 5.0e-3
t = 0.0
dt = 1.0e-3
nt = 2000
u = solve_ode(
    f,
    u,
    dt,
    nt;
    ncallback = 20,
    callback = (u, t, i) -> begin
        title = @sprintf("Solution, t = %.3f", t)
        fig = plot(x, u; xlabel = "x", title)
        display(fig)
        sleep(0.01) # Time for plot
    end,
    ν,
)

# This is a typical for the Burgers equation: The initial conditions merge to
# for a shock, which may be dampened depending on the viscosity.

# ## Discrete filtering and large eddy simulation (LES)
#
# We now assume that we are only interested in the large scale structures of the
# flow. To compute those, we would ideally like to use a coarser resolution
# ($N_\text{LES}$) than the one needed to resolve all the features of the flow
# ($N_\text{DNS}$).
#
# To define "large scales", we consider a discrete filter $\phi \in
# \mathbb{R}^{N_\text{LES} \times N_\text{DNS}}$, averaging multiple DNS grid
# points into LES points. The filtered velocity field is defined by $\bar{u} =
# \phi u$. It is governed by the equation
#
# $$
# \frac{\mathrm{d} \bar{u}}{\mathrm{d} t} = \bar{f}(\bar{u}) + c(u, \bar{u}),
# $$
#
# where $\bar{f}$ is the same as $f$ but on the coarse grid $\bar{x} = \left(
# \frac{n}{N_\text{LES}} \right)_{n = 1}^{N_\text{LES}$ and $c(u, \bar{u}) =
# \phi f(u) - \bar{f}(\bar{u})$ is the commutator error between the coarse grid
# and filtered fine grid right hand sides. To close the equations, we
# approximate the unknown commutator error using a closure model $m$ with
# parameters $\theta$:
#
# $$
# m(\bar{u}, \theta) \approx c(u, \bar{u}).
# $$
#
# We thus need to make two choices: $m$ and $\theta$. We can then solve the
# LES equation
#
# $$
# \frac{\mathrm{d} \bar{v}}{\mathrm{d} t} = \bar{f}(\bar{v}) + m(\bar{v}, θ),
# $$
#
# where the LES solution $\bar{v}$ is an approximation to the filtered DNS
# solution $\bar{u}$.
#
# The following right hand side function includes the correction term.

g(u; ν, m, θ) = f(u; ν) + m(u, θ)

# ### Model architecture
#
# We are free to choose the model architecture $m$. Here, we will consider two
# neural network architectures.

#-

# #### Fourier neural operator architecture
#
# Now let's implement the Fourier Neural Operator (FNO) [^3]. It is a network
# composed of _Fourier Layers_.
# A Fourier layer $u \mapsto w$ transforms the _function_ $u$ into the _function_ $w$.
# It is defined by the following expression in continuous physical space:
#
# $$
# w(x) = \sigma \left( z(x) + W u(x) \right), \quad \forall x \in \Omega,
# $$
#
# where $z$ is defined by its Fourier series coefficients $\hat{z}(k) = R(k)
# \hat{u}(k)$ for all wave numbers $k \in \mathbb{Z}$ and some weight matrix
# collection $R(k) \in \mathbb{C}^{n_\text{out} \times n_\text{in}}$. The
# important part is the following choice: $R(k) = 0$ for $\| k \| >
# k_\text{max}$ for some $k_\text{max}$. This truncation makes the FNO
# applicable to any spatial $N$-discretization of $u$ and $w$ as long as $N > 2
# k_\text{max}$. The same weight matrices may be reused for different
# discretizations.
#
# The deep learning framework [Lux](https://lux.csail.mit.edu/) lets us define
# our own layer types. All functions should be pure ("functional programming"),
# meaning that the same inputs should produce the same outputs. In particular,
# this also applies to random number generation and state modification. The
# weights are stored in a vector outside the layer, while the layer itself
# contains information for weight initialization.

struct FourierLayer{A,F} <: Lux.AbstractExplicitLayer
    kmax::Int
    cin::Int
    cout::Int
    σ::A
    init_weight::F
end

FourierLayer(kmax, ch::Pair{Int,Int}; σ = identity, init_weight = glorot_uniform_64) =
    FourierLayer(kmax, first(ch), last(ch), σ, init_weight)

# We also need to specify how to initialize the parameters and states. The
# Fourier layer does not have any hidden states that are modified.

Lux.initialparameters(rng::AbstractRNG, (; kmax, cin, cout, init_weight)::FourierLayer) = (;
    spatial_weight = init_weight(rng, cout, cin),
    spectral_weights = init_weight(rng, kmax + 1, cout, cin, 2),
)
Lux.initialstates(::AbstractRNG, ::FourierLayer) = (;)
Lux.parameterlength((; kmax, cin, cout)::FourierLayer) =
    cout * cin + (kmax + 1) * 2 * cout * cin
Lux.statelength(::FourierLayer) = 0

# We now define how to pass inputs through Fourier layer. In matrix notation,
# multiple samples can be processed at the same time. We therefor assume the
# following:
#
# - Input size: `(nx, cin, nsample)`
# - Output size: `(nx, cout, nsample)`

function ((; kmax, cout, cin, σ)::FourierLayer)(x, params, state)
    nx = size(x, 1)

    ## Destructure params
    ## The real and imaginary parts of R are stored in two separate channels
    W = params.spatial_weight
    W = reshape(W, 1, cout, cin)
    R = params.spectral_weights
    R = selectdim(R, 4, 1) .+ im .* selectdim(R, 4, 2)

    ## Spatial part (applied point-wise)
    y = reshape(x, nx, 1, cin, :)
    y = sum(W .* y; dims = 3)
    y = reshape(y, nx, cout, :)

    ## Spectral part (applied mode-wise)
    ##
    ## Steps:
    ##
    ## - go to complex-valued spectral space
    ## - chop off high wavenumbers
    ## - multiply with weights mode-wise
    ## - pad with zeros to restore original shape
    ## - go back to real valued spatial representation
    ikeep = 1:kmax+1
    nkeep = kmax + 1
    z = fft(x, 1)
    z = z[ikeep, :, :]
    z = reshape(z, nkeep, 1, cin, :)
    z = sum(R .* z; dims = 3)
    z = reshape(z, nkeep, cout, :)
    z = vcat(z, zeros(nx - kmax - 1, size(z, 2), size(z, 3)))
    z = real.(ifft(z, 1))

    ## Outer layer: Activation over combined spatial and spectral parts
    ## Note: Even though high wavenumbers are chopped off in `z` and may
    ## possibly not be present in the input at all, `σ` creates new high
    ## wavenumbers. High wavenumber functions may thus be represented using a
    ## sequence of Fourier layers. In this case, the `y`s are the only place
    ## where information contained in high input wavenumbers survive in a
    ## Fourier layer.
    v = σ.(y .+ z)

    ## Fourier layer does not modify state
    v, state
end

# We will use four Fourier layers, with a final dense layer.

## Number of channels
ch_fno = [2, 5, 5, 5, 5]

## Cut-off wavenumbers
kmax_fno = [8, 8, 8, 8]

## Fourier layer activations
σ_fno = [gelu, gelu, gelu, identity]

## Model
_fno = Chain(
    ## Create channel
    u -> reshape(u, size(u, 1), 1, size(u, 2)),

    ## Augment channels
    u -> cat(u, u .^ 2; dims = 2),

    ## Some Fourier layers
    (
        FourierLayer(kmax_fno[i], ch_fno[i] => ch_fno[i+1]; σ = σ_fno[i]) for
        i ∈ eachindex(σ_fno)
    )...,

    ## Put channels in first dimension
    u -> permutedims(u, (2, 1, 3)),

    ## Compress with a final dense layer
    Dense(ch_fno[end] => 2 * ch_fno[end], gelu),
    Dense(2 * ch_fno[end] => 1; use_bias = false),

    ## Put channels back after spatial dimension
    u -> permutedims(u, (2, 1, 3)),

    ## Remove singleton channel
    u -> reshape(u, size(u, 1), size(u, 3)),
)

# Create parameter vector and empty state

θ_fno, state_fno = Lux.setup(rng, _fno)
θ_fno = ComponentArray(θ_fno)
length(θ_fno)

# We also define a convenience wrapper hiding the (empty) state manipulation

fno(v, θ) = first(_fno(v, θ, state_fno))

# #### Convolutional neural network
#
# In addition to the FNO, we will use a CNN closure model. There should be fewer
# parameters, but this model only uses local spatial information.

## Kernel radii
r_cnn = [2, 2, 2, 2]

## Channels
ch_cnn = [2, 8, 8, 8, 1]

## Activations
σ_cnn = [leakyrelu, leakyrelu, leakyrelu, identity]

## Use bias
b_cnn = [true, true, true, false]

_cnn = Chain(
    ## Create singleton channel
    u -> reshape(u, size(u, 1), 1, size(u, 2)),

    ## Add a square channel (Burgers RHS has a square term, so maybe the closure
    ## model can make use of the same "structure")
    u -> cat(u, u .^ 2; dims = 2),

    ## Add padding so that output has same shape as commutator error
    u -> pad_circular(u, sum(r_cnn)),

    ## Some convolutional layers
    (
        Conv(
            (2 * r_cnn[i] + 1, 2 * r_cnn[i] + 1),
            ch_cnn[i] => ch_cnn[i+1],
            σ_cnn[i];
            use_bias = b_cnn[i],
            init_weight = glorot_uniform_64,
        ) for i ∈ eachindex(r_cnn)
    )...,

    ## Remove singleton output channel
    u -> reshape(u, size(u, 1), size(u, 3)),
)

# Create parameter vector and empty state

θ_cnn, state_cnn = Lux.setup(rng, _cnn)
θ_cnn = ComponentArray(θ_cnn)
length(θ_cnn)

# Convenience wrapper

cnn(v, θ) = first(_cnn(v, θ, state_cnn))

# ### Choosing model parameters: loss function
#
# To choose $\theta$, we will minimize a loss function ("train" the neural
# network). Since the model predicts the commutator error, the obvious choice is
# the a priori loss function
#
# $$
# L^\text{prior}(\theta) = \| m(\bar{\hat{u}}, \theta) - c(\hat{u}, \bar{\hat{u}}) \|^2.
# $$
#
# This loss function has a simple computational chain, that is mostly comprised
# of evaluating the neural network $\theta$ itself. Computing the derivative
# with respect to $\theta$ is thus simple. We call this function "a priori"
# since it only measures the error of the prediction itself, and not the effect
# this error has on the LES solution $\bar{v}$.
#
# Let's start with the a priori loss. Since instability is not directly
# detected, we add a regularization term to penalize extremely large weights.

mean_squared_error(f, x, y, θ; λ = 1.0e-8) =
    sum(abs2, f(x, θ) - y) / sum(abs2, y) + λ * sum(abs2, θ) / length(θ)

# We will only use a subset (`nuse`) of all (`ntrain * nt`) the samples for each
# loss evaluation.

function create_randloss_commutator(f, u, c; nuse = 20)
    u = reshape(u, size(x, 1), :)
    c = reshape(c, size(y, 1), :)
    nsample = size(u, 2)
    function randloss(θ)
        i = Zygote.@ignore sort(shuffle(1:nsample)[1:nuse])
        uuse = Zygote.@ignore u[:, i]
        cuse = Zygote.@ignore c[:, i]
        mean_squared_error(f, uuse, cuse, θ)
    end
end

# Ideally, we want the LES simulation to produce the filtered DNS velocity
# $\bar{\hat{u}}$. We can thus alternatively minimize the a posteriori loss
# function
#
# $$
# L^\text{post}(\theta) = \| \bar{\hat{v}}_\theta - \bar{\hat{u}} \|^2.
# $$
#
# This loss function contains more information about the effect of $\theta$ than
# $L^\text{prior}$. However, it is comprised of a significantly longer
# computational chain, as it includes time stepping in addition to the neural
# network evaluation itself. Computing the gradient with respect to $\theta$ is
# thus more costly, and also requires an AD-friendly time stepping scheme (which
# we have already taken care of above). Note that it is also possible to compute
# the gradient of the time-continuous ODE instead of the time-discretized one as
# we do here. It involves solving an adjoint ODE backwards in time, which in
# turn has to be discretized. Our approach here is therefore called
# "discretize-then-optimize", while the adjoint ODE method is called
# "optimize-then-discretize". The [SciML](https://github.com/sciml) time
# steppers include both methods.
#
# For the a posteriori loss function, we provide the right hand side function
# `f` (including closure), filtered DNS time series `ubar`, and model
# parameters. We compute the error between the predicted and reference trajectories
# at each time point.

function trajectory_loss(f, ubar; dt = 1.0e-3, params...)
    nt = size(ubar, 3)
    loss = 0.0
    v = ubar[:, :, 1]
    for i = 2:nt
        v = step_rk4(f, v, dt; params...)
        u = ubar[:, :, i]
        loss += sum(abs2, v - u) / sum(abs2, u)
    end
    loss
end

# To limit the length of the computational chain, we only unroll `nunroll`
# time steps at each loss evaluation. The time step from which to unroll is
# chosen at random at each evaluation, as are the initial conditions.

function create_randloss_trajectory(f, ubar; dt, nuse = 1, nunroll = 10, params...)
    nsample = size(ubar, 2)
    nt = size(ubar, 3)
    function randloss(θ)
        isample = Zygote.@ignore sort(shuffle(1:nsample)[1:nuse])
        istart = Zygote.@ignore rand(1:nt-nunroll)
        it = Zygote.@ignore istart:istart+nunroll
        trajectory = Zygote.@ignore ubar[:, isample, it]
        trajectory_loss(f, trajectory; dt, params..., θ)
    end
end

# ### Data generation
#
# Now we set up the experiment. We need to decide on the following:
#
# - Problem parameter: $\nu$
# - LES resolution
# - DNS resolution
# - Discrete filter
# - Number of initial conditions
# - Closure model: FNO and CNN
# - Simulation time: Too short, and we won't have time to detect long-term
#   instabilities from our model; too long, and most of the data will be too
#   smooth for a closure model to be needed (due to viscosity)
#
# In addition, we will split our data into
#
# - Training data (for choosing $\theta$)
# - Validation data (just for monitoring training, choose when to stop)
# - Testing data (for testing performance on unseen data)

## Viscosity
ν = 5e-3

## Resolution
n_les = 64
n_dns = 256

## Grids
x_les = LinRange(0.0, 1.0, n_les + 1)[2:end]
x_dns = LinRange(0.0, 1.0, n_dns + 1)[2:end]

## Grid sizes
Δ_les = 1 / n_les
Δ_dns = 1 / n_dns

# Create filter

## Filter width
Δϕ = 3Δ_les

## Filter kernel
gaussian(Δ, x) = sqrt(6 / π) / Δ * exp(-6x^2 / Δ^2)
top_hat(Δ, x) = (abs(x) ≤ Δ / 2) / Δ
kernel = gaussian

## Discrete filter matrix (with periodic extension and threshold for sparsity)
ϕ = sum(-1:1) do z
    d = x_les .- x_dns' .- z
    kernel.(Δϕ, d) .* (abs.(d) .≤ 3 ./ 2 .* Δϕ)
end
ϕ = ϕ ./ sum(ϕ; dims = 2) ## Normalize weights
ϕ = sparse(ϕ)
dropzeros!(ϕ)
heatmap(ϕ; yflip = true, xmirror = true, title = "Filter")

# Create initial conditions

## Number of initial conditions
ntrain = 10
nvalid = 2
ntest = 5

## Absolute positions
itrain = 1:ntrain
ivalid = ntrain+1:ntrain+nvalid
itest = ntrain+nvalid+1:ntrain+nvalid+ntest

## Initial conditions
u = create_initial_conditions(
    x_dns,
    ntrain + nvalid + ntest;
    kmax = 10,
    decay = k -> 1 / (1 + abs(k))^1.2,
)

# DNS time stepping

t = 0.0
dt = 1.0e-3
nt = 1000

## Filtered snapshots (including at t = 0)
v = zeros(n_les, ntrain + nvalid + ntest, nt + 1)

## Commutator errors (including at t = 0)
c = zeros(n_les, ntrain + nvalid + ntest, nt + 1)

## Save filtered solution and commutator error after each DNS time step
function save_sol(u, t, i)
    ubar = ϕ * u
    v[:, :, i+1] = ubar
    c[:, :, i+1] = ϕ * f(u; ν) - f(ubar; ν)
    if i % 10 == 0
        title = @sprintf("Solution, t = %.3f", t)
        fig = plot(; xlabel = "x", title)
        plot!(fig, x_dns, u[:, 3]; linestyle = :dash, label = "u")
        plot!(fig, x_les, ubar[:, 3]; label = "ubar")
        display(fig)
        sleep(0.001) # Time for plot
    end
end

## Save for t = 0 first
save_sol(u, t, 0)

## Do time stepping
solve_ode(f, u, dt, nt; callback = save_sol, ncallback = 1, ν)

## Split data
v_train, c_train = v[:, :, itrain], c[:, :, itrain]
v_valid, c_valid = v[:, :, ivalid], c[:, :, ivalid]
v_test, c_test = v[:, :, itest], c[:, :, itest]

# Choose closure model

m, θ₀ = fno, θ_fno
## m, θ₀ = cnn, θ_cnn

# Choose loss function

# ### Training
#
# First, we choose a loss function.

randloss = create_randloss_commutator(m, v_train, c_train; nuse = 50)
## randloss = create_randloss_trajectory(g, v_train; dt = 1f-3, nuse = 3, nunroll = 10, ν)

# Model warm-up: trigger compilation and get indication of complexity

randloss(θ₀)
gradient(randloss, θ₀);
@time randloss(θ₀);
@time gradient(randloss, θ₀);

# We will monitor the error along the way.

θ = θ₀
opt = Optimisers.setup(Adam(1.0e-3), θ)
ncallback = 20
niter = 1000
ihist = Int[]
ehist_prior = zeros(0)
ehist_post = zeros(0)
ishift = 0

# The cell below can be repeated to continue training. It plots the relative a
# priori and a posteriori errors on the validation set

for i = 1:niter
    ∇ = first(gradient(randloss, θ))
    opt, θ = Optimisers.update(opt, θ, ∇)
    println(i)
    if i % ncallback == 0
        vv, cc = reshape(v_valid, n_les, :), reshape(c_valid, n_les, :)
        e_prior = norm(m(vv, θ) - cc) / norm(cc)
        vθ = solve_ode(g, v_valid[:, :, 1], dt, nt; ν, m, θ)
        vnm = solve_ode(f, v_valid[:, :, 1], dt, nt; ν)
        e_post = norm(vθ - v_valid[:, :, end]) / norm(v_valid[:, :, end])
        enm = norm(vnm - v_valid[:, :, end]) / norm(v_valid[:, :, end])
        push!(ihist, ishift + i)
        push!(ehist_prior, e_prior)
        push!(ehist_post, e_post)
        fig = plot(; xlabel = "Iterations", title = "Relative error")
        hline!(fig, [1.0]; color = 1, linestyle = :dash, label = "A priori: No model")
        plot!(fig, ihist, ehist_prior; color = 1, label = "A priori: FNO")
        hline!(fig, [enm]; color = 2, linestyle = :dash, label = "A posteriori: No model")
        plot!(fig, ihist, ehist_post; color = 2, label = "A posteriori: FNO")
        display(fig)
    end
end
ishift += niter

# ### Model performance
#
# We will now make a comparison of the three closure models (including the
# "no-model" where $m = 0$, which correspond to solving the DNS equations on the
# LES grid).

# ## References
#
# [^3]: Z. Li, N. Kovachki, K. Azizzadenesheli, B. Liu, K. Bhattacharya, A. Stuart, and
#       A. Anandkumar.  _Fourier neural operator for parametric partial differential
#       equations._ arXiv:[2010.08895](https://arxiv.org/abs/2010.08895), 2021.
