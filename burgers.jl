# # Neural closure models for the viscous Burgers equation
#
# In this tutorial, we will train neural closure models for the viscous Burgers
# equation in the [Julia](https://julialang.org/) programming language. We here
# use Julia for ease of use, efficiency, and writing differentiable code to
# power scientific machine learning. This file is available as a commented
# Julia script, or as a Jupyter notebook.

#-

# ## Running locally
#
# 1. Install Julia from one of the following:
#    - the official [downloads page](https://julialang.org/downloads/) (select
#      the binary for your platform)
#    - the [Juliaup](https://github.com/JuliaLang/juliaup) version manager.
#      This is the preferred way, as you will get notified about updates to
#      Julia. It requires typing a line into your command line (see README).
# 2. Install [VSCode](https://code.visualstudio.com/)
# 3. Install the Julia [extension](https://code.visualstudio.com/docs/languages/julia) for VSCode.
#
# In VSCode, you can then choose one of the two options:
#
# - Open the `burgers.jl` file, and execute it line by line with `Shift` +
#   `Enter`. A Julia REPL should open up, and plots should appear in a separate
#   pane.
# - Open the notebook version `burgers.ipynb`. Execute cell by cell with
#   `Shift` + `Enter`. The output and plots appear below each cell. You also
#   get to see the rendered LaTeX equations.
#
# If you do not want to install Julia locally, you can run it on a Google
# cloud machine instead (see next section).

#-

#nb # ## Running on Google Colab
#nb #
#nb # _This section is only needed when running on Google colab._
#nb #
#nb # To use Julia on Google colab, we will install Julia using the official version
#nb # manager Juliup. From the default Python kernel, we can access the shell by
#nb # starting a line with `!`.
#nb 
#nb !curl -fsSL https://install.julialang.org | sh -s -- --yes
#nb 
#nb # We can check that Julia is successfully installed on the Colab instance.
#nb 
#nb !~/.juliaup/bin/julia -e 'println("Hello")'
#nb 
#nb # We now proceed to install the necessary Julia packages, including `IJulia` which
#nb # will add the Julia notebook kernel.
#nb 
#nb %%shell
#nb ~/.juliaup/bin/julia -e '''
#nb     using Pkg
#nb     Pkg.add([
#nb         "ComponentArrays",
#nb         "FFTW",
#nb         "IJulia",
#nb         "LinearAlgebra",
#nb         "Lux",
#nb         "NNlib",
#nb         "Optimisers",
#nb         "Plots",
#nb         "Printf",
#nb         "Random",
#nb         "SparseArrays",
#nb         "Zygote",
#nb     ])
#nb '''
#nb
#nb # Once this is done, do the following:
#nb #
#nb # 1. Reload the browser page (`CTRL`/`CMD` + `R`)
#nb # 2. In the top right corner of Colab, then select the Julia kernel.
#nb #
#nb # ![](https://github.com/agdestein/NeuralNavierStokes/blob/main/assets/select.png?raw=true)
#nb # ![](https://github.com/agdestein/NeuralNavierStokes/blob/main/assets/runtime.png?raw=true)

#nb #-

# ## Preparing the simulations
#
# Julia comes with built in array functionality. Additional functionality is
# provided in various packages, some of which are available in the Standard
# Library (LinearAlgebra, Printf, Random, SparseArrays). Others are available in
# the General Registry, and can be added using the built in package manager Pkg,
# e.g. `using Pkg; Pkg.add("Plots")`. If you ran the Colab setup section, the
# packages should already be added.

## using Pkg
## Pkg.add([
##     "ComponentArrays",
##     "FFTW",
##     "LinearAlgebra",
##     "Lux",
##     "NNlib",
##     "Optimisers",
##     "Plots",
##     "Printf",
##     "Random",
##     "SparseArrays",
##     "Zygote",
## ])

#-

using ComponentArrays
using FFTW
using LinearAlgebra
using Lux
using NNlib
using Optimisers
using Plots
using Printf
using Random
using SparseArrays
using Zygote

# The deep learning framework [Lux](https://lux.csail.mit.edu/) likes to toss
# random number generators (RNGs) around, for reproducible science. We
# therefore need to initialize an RNG. The seed makes sure the same sequence of
# pseudo-random numbers are generated at each time the session is restarted.

Random.seed!(123)
rng = Random.default_rng()

# Deep learning functions usually use single precision floating point numbers
# by default, as this is preferred on GPUs. Julia itself, on the other hand, is
# slightly opinionated towards double precision floating point numbers, e.g.
# `3.14`, `1 / 6` and `2π` are all of type `Float64` (their single precision
# counterparts would be the slightly more verbose `3.14f0`, `1f0 / 6` (or
# `Float32(1 / 6)`) and `2f0π`). For simplicity, we will only use `Float64`.
# The only place we will encounter `Float32` in this file is in the default
# neural network weight initializers, so here is an alternative weight
# initializer using double precision.

glorot_uniform_64(rng::AbstractRNG, dims...) = glorot_uniform(rng, Float64, dims...)

# ## The viscous Burgers equation
#
# Consider the periodic domain $\Omega = [0, 1]$. The visous Burgers equation
# is given by
#
# $$
# \frac{\partial u}{\partial t} = - \frac{1}{2} \frac{\partial }{\partial x}
# \left( u^2 \right) + \nu \frac{\partial^2 u}{\partial x^2},
# $$
#
# where $\nu > 0$ is the viscosity.
#
# ### Discretization
#
# Consider a uniform discretization $x = \left( \frac{n}{N} \right)_{n =
# 1}^N$, with the additional point $x_0 = 0$ overlapping with $x_N$. The step
# size is $\Delta x = \frac{1}{N}$. Using a
# central finite difference, we get the discrete equations
#
# $$
# \frac{\mathrm{d} u_n}{\mathrm{d} t} = - \frac{1}{2} \frac{u_{n + 1}^2 - u_{n -
# 1}^2}{2 \Delta x} + \nu \frac{u_{n + 1} - 2 u_n + u_{n - 1}}{\Delta x^2},
# $$
#
# with the convention $u_0 = u_N$ and $u_{N + 1} = u_1$ (periodic extension). The
# degrees of freedom are stored in the vector $u = (u_n)_{n = 1}^N$. In vector
# notation, we will write this as $\frac{\mathrm{d} u}{\mathrm{d} t} = f(u)$.
# Solving this equation for sufficiently small $\Delta x$ (large $N$) will be
# referred to as _direct numerical simulation_ (DNS), and can be expensive.
#
# Note: This is a simple discretization, not ideal for dealing with shocks.

#-

# We start by defining the right hand side function `f` for a vector `u`, making
# sure to account for the periodic boundaries. The macro `@.` makes sure that all
# following operations are performed element-wise. Note that `circshift` here
# acts along the first dimension, so `f` can be applied to multiple snapshots
# at once (stored as columns in the matrix `u`).

function f(u; ν)
    N = size(u, 1)
    u₊ = circshift(u, -1)
    u₋ = circshift(u, 1)
    @. -N / 4 * (u₊^2 - u₋^2) + N^2 * ν * (u₊ - 2u + u₋)
end

# ### Time discretization
#
# For time stepping, we do a simple explicit Runge-Kutta scheme (RK).
#
# From a current state $u^0 = u(t)$, we divide the outer time step
# $\Delta t$ into $i \in \{1, \dots, s\}$ sub-steps as follows:
#
# $$
# \begin{split}
# F^i & = f(u^{i - 1}) \\
# u^i & = u^0 + \Delta t \sum_{j = 1}^{i} A_{i j} F^j,
# \end{split}
# $$
#
# where $A \in \mathbb{R}^{s \times s}$ are the coefficients of the RK method.
# The solution at the next outer time step $t + \Delta t$ is then
# $u^s = u(t + \Delta t) + \mathcal{O}(\Delta t^r)$ where $r$ is the order of
# the RK method.
# A fourth order method is given by the following coefficients ($s = 4$, $r =
# 4$):
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
# The following function performs one RK4 time step. Note that we never modify
# any vectors, only create new ones. The AD-framework Zygote prefers it this
# way. The syntax `params...` lets us pass a variable number of keyword
# arguments to the right hand side function `f` (for the above `f` there is
# only one: `ν`).

function step_rk4(f, u₀, dt; params...)
    A = [
        1 0 0 0
        0 1 0 0
        0 0 1 0
        1/6 2/6 2/6 1/6
    ]
    u = u₀
    k = ()
    for i = 1:size(A, 1)
        ki = f(u; params...)
        k = (k..., ki)
        u = u₀
        for j = 1:i
            u += dt * A[i, j] * k[j]
        end
    end
    u
end

# Solving the ODE is done by chaining individual time steps. We here add the
# option to call a callback function after each time step. Note that the path
# to the final output `u` is obtained by passing the inputs `u₀` and
# parameters `params` through a finite amount of computational steps, each of
# which should have a chain rule defined and recognized in the Zygote AD
# framework. Solving the ODE should be differentiable, as long as `f` is.

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

function create_initial_conditions(
    nx,
    nsample;
    kmax = 10,
    decay = k -> 1 / (1 + abs(k))^1.2,
)
    ## Fourier basis
    basis = [exp(2π * im * k * x / nx) for x ∈ 1:nx, k ∈ -kmax:kmax]

    ## Fourier coefficients with random phase and amplitude
    c = [randn() * exp(-2π * im * rand()) * decay(k) for k ∈ -kmax:kmax, _ ∈ 1:nsample]

    ## Random data samples (real-valued)
    real.(basis * c)
end

# ### Example simulation
#
# Let's test our method in action.

ν = 1.0e-3
nx = 256
x = LinRange(0.0, 1.0, nx + 1)[2:end]

## Initial conditions (one sample vector)
u = create_initial_conditions(nx, 1)

## Time stepping
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

# This is typical for the Burgers equation: The initial conditions merge to
# a shock, which may be dampened depending on the viscosity.

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
# \frac{\mathrm{d} \bar{u}}{\mathrm{d} t} = f(\bar{u}) + c(u, \bar{u}),
# $$
#
# where $f$ is adapted to the grid of its input field ($\bar{u}$) and
# $c(u, \bar{u}) = \overline{f(u)} - f(\bar{u})$ is the commutator error
# between the coarse grid and filtered fine grid right hand sides. To close the
# equations, we approximate the unknown commutator error using a closure model
# $m$ with parameters $\theta$:
#
# $$
# m(\bar{u}, \theta) \approx c(u, \bar{u}).
# $$
#
# We thus need to make two choices: $m$ and $\theta$. While the model $m$ will
# be chosen based on our expertise of numerical methods and machine learning,
# the parameters $\theta$ will be fitted using high fidelity simulation data.
# We can then solve the LES equation
#
# $$
# \frac{\mathrm{d} \bar{v}}{\mathrm{d} t} = f(\bar{v}) + m(\bar{v}, θ),
# $$
#
# where the LES solution $\bar{v}$ is an approximation to the filtered DNS
# solution $\bar{u}$.
#
# The following right hand side function includes the correction term.

g(u; ν, m, θ) = f(u; ν) + m(u, θ)

# ### Data generation
#
# This generic function creates a data structure containing filtered DNS data,
# commutator errors and simulation parameters for a given filter $ϕ$. We also
# provide some default values.

function create_data(
    nsample,
    ϕ;
    dt = 1.0e-3,
    nt = 1000,
    kmax = 10,
    decay = k -> 1 / (1 + abs(k))^1.2,
    ν = 1e-3,
)
    ## Resolution
    nx_les, nx_dns = size(ϕ)

    ## Grids
    x_les = LinRange(0.0, 1.0, nx_les + 1)[2:end]
    x_dns = LinRange(0.0, 1.0, nx_dns + 1)[2:end]

    ## Output data
    data = (;
        ## Filtered snapshots and commutator errors (including at t = 0)
        u = zeros(nx_les, nsample, nt + 1),
        c = zeros(nx_les, nsample, nt + 1),

        ## Simulation-specific parameters
        dt,
        ν,
    )

    ## DNS Initial conditions
    u = create_initial_conditions(nx_dns, nsample; kmax, decay)

    ## Save filtered solution and commutator error after each DNS time step
    function callback(u, t, i)
        ubar = ϕ * u
        data.u[:, :, i+1] = ubar
        data.c[:, :, i+1] = ϕ * f(u; ν) - f(ubar; ν)
        if i % 10 == 0
            title = @sprintf("Solution, t = %.3f", t)
            np = min(3, nsample)
            fig = plot(; xlabel = "x", title)
            plot!(fig, x_dns, u[:, 1:np]; color = (1:np)', linestyle = :dash, label = "u")
            plot!(fig, x_les, ubar[:, 1:np]; color = (1:np)', label = "ubar")
            display(fig)
            sleep(0.001) # Time for plot
        end
    end

    ## Save for t = 0.0 first
    callback(u, 0.0, 0)

    ## Do time stepping (save after each step)
    solve_ode(f, u, dt, nt; callback, ncallback = 1, ν)

    ## Return data
    data
end

# ### Model architecture
#
# We are free to choose the model architecture $m$. Here, we will consider two
# neural network architectures. The following wrapper returns the model and
# initial parameters for a Lux `Chain`. Note: If the chain includes
# state-dependent layers such as `Dropout` (which modify their RNGs at each
# evaluation), this wrapper should not be used.

function create_model(chain, rng)
    ## Create parameter vector and empty state
    θ, state = Lux.setup(rng, chain)

    ## Convert nested named tuples of arrays to a ComponentArray,
    ## which behaves like a long vector
    θ = ComponentArray(θ)

    ## Convenience wrapper for empty state in input and output
    m(v, θ) = first(chain(v, θ, state))

    ## Return model and initial parameters
    m, θ
end

# #### Convolutional neural network architecture (CNN)
#
# A CNN is an interesting closure models for the following reasons:
#
# - The parameters are sparse, since the kernels are reused for each output
# - A convolutional layer can be seen as a discretized differential operator
# - Translation invariance, a desired physical property of the commutator
#   error we try to predict, is baked in.
#
# However, it only uses local spatial information, whereas an ideal closure
# model could maybe recover some of the missing information in far-away values.
#
# Note that we start by adding input channels, stored in a tuple of functions.
# The Burgers RHS has a square term, so maybe the closure model can make use of
# the same "structure" [^4].

"""
    create_cnn(; r, channels, σ, use_bias, rng, input_channels = (u -> u, u -> u .^ 2))

Create CNN.

Keyword arguments:

- `r`: Vector of kernel radii
- `channels`: Vector layer output channel numbers
- `σ`: Vector of activation functions
- `use_bias`: Vectors of indicators for using bias
- `rng`: Random number generator
- `input_channels`: Tuple of input channel contstructors

Return `(cnn, θ)`, where `cnn(v, θ)` acts like a force on `v`.
"""
function create_cnn(; r, channels, σ, use_bias, rng, input_channels = (u -> u, u -> u .^ 2))
    @assert channels[end] == 1 "A unique output channel is required"

    ## Add number of input channels
    channels = [length(input_channels); channels]

    ## Padding length
    padding = sum(r)

    ## Create CNN
    create_model(
        Chain(
            ## Create singleton channel
            u -> reshape(u, size(u, 1), 1, size(u, 2)),

            ## Create input channels
            u -> hcat(map(i -> i(u), input_channels)...),

            ## Add padding so that output has same shape as commutator error
            u -> pad_circular(u, padding; dims = 1),

            ## Some convolutional layers
            (
                Conv(
                    (2 * r[i] + 1,),
                    channels[i] => channels[i+1],
                    σ[i];
                    use_bias = use_bias[i],
                    init_weight = glorot_uniform_64,
                ) for i ∈ eachindex(r)
            )...,

            ## Remove singleton output channel
            u -> reshape(u, size(u, 1), size(u, 3)),
        ),
        rng,
    )
end

# #### Fourier neural operator architecture (FNO)
#
# Let's implement the FNO [^3]. It is a network composed of _Fourier Layers_
# (FL). A Fourier layer $u \mapsto w$ transforms the _function_ $u$ into the
# _function_ $w$. It is defined by the following expression in continuous
# physical space:
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
# Note that a standard convolutional layer (CL) can also be written in spectral
# space, where the spatial convolution operation becomes an wave number
# element-wise product. The effective difference between the layers of a FNO
# and CNN becomes the following:
#
# - A FL does not include the bias of a CL
# - The spatial part of a FL corresponds to the central weight of the CL
#   kernel
# - A CL is chosen to be sparse in physical space (local kernel with radius $r
#   \ll N / 2$), and would therefore be dense in spectral space ($k_\text{max} =
#   N / 2$)
# - The spectral part of a FL is chosen to be sparse in spectral space
#   ($k_\text{max} \ll N / 2$), and would therefore dense in physical space (it
#   can be written as a convolution stencil with radius $r = N / 2$)
# 
# Lux lets us define
# our own layer types. All functions should be "pure" (functional programming),
# meaning that the same inputs should produce the same outputs. In particular,
# this also applies to random number generation and state modification. The
# weights are stored in a vector outside the layer, while the layer itself
# contains information for weight initialization.
#
# If you are not familiar with Julia, feel free to go quickly through
# the following code cells. Just note that all variables have a type (e.g.
# `kmax::Int` means that `kmax` is an integer), but most of the time we don't have
# to declare types explicitly. Structures (`struct`s) can be
# parametrized and specialized for the types we give them in the constructor
# (e.g. `σ::A` means that a specialized version of the struct is compiled for
# each activation function we give it, creating an optimized FourierLayer for
# `σ = relu` where `A = typeof(relu)`, and a different version optimized for `σ
# = tanh` where `A = typeof(tanh)` etc.). Here our layer will have the type
# `FourierLayer`, with a default and custom constructor (two constructor
# methods, the latter making use of the default).

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
# Fourier layer does not have any hidden states that are modified. The below
# code adds methods to some existing Lux functions. These new methods are only
# used when the functions encounter `FourierLayer` inputs. For example, in our
# current environment, we have this many methods for the function
# `Lux.initialparameters` (including `Dense`, `Conv`, etc.):

length(methods(Lux.initialparameters))

# Now, when we add our own method, there should be one more in the method
# table.

Lux.initialparameters(rng::AbstractRNG, (; kmax, cin, cout, init_weight)::FourierLayer) = (;
    spatial_weight = init_weight(rng, cout, cin),
    spectral_weights = init_weight(rng, kmax + 1, cout, cin, 2),
)
Lux.initialstates(::AbstractRNG, ::FourierLayer) = (;)
Lux.parameterlength((; kmax, cin, cout)::FourierLayer) =
    cout * cin + (kmax + 1) * 2 * cout * cin
Lux.statelength(::FourierLayer) = 0

## Pretty printing
function Base.show(io::IO, (; kmax, cin, cout, σ)::FourierLayer)
    print(io, "FourierLayer(", kmax)
    print(io, ", ", cin, " => ", cout)
    print(io, "; σ = ", σ)
    print(io, ")")
end

## One more method now
length(methods(Lux.initialparameters))

# This is one of the advantages of Julia: As users we can extend functions from
# other authors without modifying their package or being forced to "inherit"
# their data structures (classes). This has created an interesting package
# ecosystem. For example, [ODE
# solvers](https://github.com/SciML/OrdinaryDiffEq.jl) can be used with funky
# number types such as `BigFloat`, dual numbers, or quaternions. [Iterative
# solvers](https://github.com/JuliaLinearAlgebra/IterativeSolvers.jl)
# work out of the box with different array types, including various [GPU
# arrays](https://github.com/JuliaGPU/GPUArrays.jl), without actually
# containing any GPU array specific code.
#
# We now define how to pass inputs through a Fourier layer. In tensor notation,
# multiple samples can be processed at the same time. We therefore assume the
# following:
#
# - Input size: `(nx, cin, nsample)`
# - Output size: `(nx, cout, nsample)`

## This makes FourierLayers callable
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
    w = σ.(z .+ y)

    ## Fourier layer does not modify state
    w, state
end

# We will chain some Fourier layers, with a final dense layer. As for the CNN,
# we allow for a tuple of predetermined input channels.

"""
    create_fno(; channels, kmax, σ, rng, input_channels = (u -> u, u -> u .^ 2))

Create FNO.

Keyword arguments:

- `channels`: Vector of output channel numbers
- `kmax`: Vector of cut-off wavenumbers
- `σ`: Vector of activation functions
- `rng`: Random number generator
- `input_channels`: Tuple of input channel contstructors

Return `(fno, θ)`, where `fno(v, θ)` acts like a force on `v`.
"""
function create_fno(; channels, kmax, σ, rng, input_channels = (u -> u, u -> u .^ 2))
    ## Add number of input channels
    channels = [length(input_channels); channels]

    ## Model
    create_model(
        Chain(
            ## Create singleton channel
            u -> reshape(u, size(u, 1), 1, size(u, 2)),

            ## Create input channels
            u -> hcat(map(i -> i(u), input_channels)...),

            ## Some Fourier layers
            (
                FourierLayer(kmax[i], channels[i] => channels[i+1]; σ = σ[i]) for
                i ∈ eachindex(kmax)
            )...,

            ## Put channels in first dimension
            u -> permutedims(u, (2, 1, 3)),

            ## Compress with a final dense layer
            Dense(channels[end] => 2 * channels[end], gelu),
            Dense(2 * channels[end] => 1; use_bias = false),

            ## Put channels back after spatial dimension
            u -> permutedims(u, (2, 1, 3)),

            ## Remove singleton channel
            u -> reshape(u, size(u, 1), size(u, 3)),
        ),
        rng,
    )
end

# ### Choosing model parameters
#
# To choose $\theta$, we will minimize a loss function using an gradient
# descent based optimization method ("train" the neural
# network).

#-

# #### Loss function
#
# Since the model predicts the commutator error, the obvious choice is
# the a priori loss function
#
# $$
# L^\text{prior}(\theta) = \| m(\bar{u}, \theta) - c(u, \bar{u}) \|^2.
# $$
#
# This loss function has a simple computational chain, that is mostly comprised
# of evaluating the neural network $m$ itself. Computing the derivative
# with respect to $\theta$ is thus simple. We call this function "a priori"
# since it only measures the error of the prediction itself, and not the effect
# this error has on the LES solution $\bar{v}$.
#
# Let's start with the a priori loss. Since instability is not directly
# detected, we add a regularization term to penalize extremely large weights.

mean_squared_error(f, x, y, θ; λ) =
    sum(abs2, f(x, θ) - y) / sum(abs2, y) + λ * sum(abs2, θ) / length(θ)

# We will only use a random subset (`nuse`) of all (`nsample * nt`)
# solution snapshots at each loss evaluation. The `Zygote.@ignore` macro just
# tells the AD engine Zygote not to complain about the random index selection.

function create_randloss_commutator(m, data; nuse = 20, λ = 1.0e-8)
    (; u, c) = data
    u = reshape(u, size(u, 1), :)
    c = reshape(c, size(c, 1), :)
    nsample = size(u, 2)
    function randloss(θ)
        i = Zygote.@ignore sort(shuffle(1:nsample)[1:nuse])
        uuse = Zygote.@ignore u[:, i]
        cuse = Zygote.@ignore c[:, i]
        mean_squared_error(m, uuse, cuse, θ; λ)
    end
end

# Ideally, we want the LES simulation to produce the filtered DNS velocity
# $\bar{u}$. We can thus alternatively minimize the a posteriori loss
# function
#
# $$
# L^\text{post}(\theta) = \| \bar{v}_\theta - \bar{u} \|^2.
# $$
#
# This loss function contains more information about the effect of $\theta$
# than $L^\text{prior}$. However, it has a significantly longer computational
# chain, as it includes time stepping in addition to the neural network
# evaluation itself. Computing the gradient with respect to $\theta$ is thus
# more costly, and also requires an AD-friendly time stepping scheme (which we
# have already taken care of above). Note that it is also possible to compute
# the gradient of the time-continuous ODE instead of the time-discretized one
# as we do here. It involves solving an adjoint ODE backwards in time, which in
# turn has to be discretized. Our approach here is therefore called
# "discretize-then-optimize", while the adjoint ODE method is called
# "optimize-then-discretize". The [SciML](https://github.com/sciml) time
# steppers include both methods, as well as useful strategies for evaluating
# them efficiently.
#
# For the a posteriori loss function, we provide the right hand side function
# `model` (including closure), reference trajectories `u`, and model
# parameters. We compute the error between the predicted and reference trajectories
# at each time point.

function trajectory_loss(model, u; dt, params...)
    nt = size(u, 3)
    loss = 0.0
    v = u[:, :, 1]
    for i = 2:nt
        v = step_rk4(model, v, dt; params...)
        ui = u[:, :, i]
        loss += sum(abs2, v - ui) / sum(abs2, ui)
    end
    loss / (nt - 1)
end

# We also make a non-squared variant for error analysis.

function trajectory_error(model, u; dt, params...)
    nt = size(u, 3)
    loss = 0.0
    v = u[:, :, 1]
    for i = 2:nt
        v = step_rk4(model, v, dt; params...)
        ui = u[:, :, i]
        loss += norm(v - ui) / norm(ui)
    end
    loss / (nt - 1)
end

# To limit the length of the computational chain, we only unroll `nunroll`
# time steps at each loss evaluation. The time step from which to unroll is
# chosen at random at each evaluation, as are the initial conditions (`nuse`).
#
# The non-trainable parameters (e.g. $\nu$) are passed in `params`.

function create_randloss_trajectory(model, data; nuse = 1, nunroll = 10, params...)
    (; u, dt) = data
    nsample = size(u, 2)
    nt = size(ubar, 3)
    function randloss(θ)
        isample = Zygote.@ignore sort(shuffle(1:nsample)[1:nuse])
        istart = Zygote.@ignore rand(1:nt-nunroll)
        it = Zygote.@ignore istart:istart+nunroll
        uuse = Zygote.@ignore u[:, isample, it]
        trajectory_loss(model, uuse; dt, params..., θ)
    end
end

# ### Training
#
# During training, we will monitor the error on the validation dataset with a
# callback. We will plot the history of the a priori and a posteriori errors.

## Initial empty history
initial_callbackstate() = (; ihist = Int[], ehist_prior = zeros(0), ehist_post = zeros(0))

## Create callback for given model and dataset
function create_callback(m, data)
    (; u, c, dt, ν) = data
    uu, cc = reshape(u, size(u, 1), :), reshape(c, size(c, 1), :)
    e_post_ref = trajectory_error(f, u; dt, ν)
    function callback(i, θ, state)
        (; ihist, ehist_prior, ehist_post) = state
        state = (;
            ihist = vcat(ihist, i),
            ehist_prior = vcat(ehist_prior, norm(m(uu, θ) - cc) / norm(cc)),
            ehist_post = vcat(ehist_post, trajectory_error(g, u; dt, ν, m, θ)),
        )
        fig = plot(; yscale = :log10, xlabel = "Iterations", title = "Relative error")
        hline!(fig, [1.0]; color = 1, linestyle = :dash, label = "A priori: No model")
        plot!(fig, state.ihist, state.ehist_prior; color = 1, label = "A priori: Model")
        hline!(
            fig,
            [e_post_ref];
            color = 2,
            linestyle = :dash,
            label = "A posteriori: No model",
        )
        plot!(fig, state.ihist, state.ehist_post; color = 2, label = "A posteriori: Model")
        display(fig)
        println("Iteration $i")
        state
    end
end

# For training, we have to initialize an optimizer and a callbackstate (lots of
# state initilization in this session)

initial_trainstate(optimiser, θ) = (;
    opt = Optimisers.setup(optimiser, θ),
    θ,
    callbackstate = initial_callbackstate(),
    istart = 0,
)

"""
    train(; loss, opt, θ, istart, niter, ncallback, callback, callbackstate)

Perform `niter` iterations of gradient descent on `loss(θ)` for given `optimiser`.
Every `ncallback` iteration, call `callback(i, θ, callbackstate)` and update
callback state, where `i` starts at `istart`.

Return a named tuple `(; opt, θ, callbcackstate)` containing the updated
optimizer state, parameters and callback state.
"""
function train(; loss, opt, θ, istart, niter, ncallback, callback, callbackstate)
    for i = 1:niter
        ∇ = first(gradient(loss, θ))
        opt, θ = Optimisers.update(opt, θ, ∇)
        i % ncallback == 0 && (callbackstate = callback(istart + i, θ, callbackstate))
    end
    istart += niter
    (; opt, θ, callbackstate, istart)
end

# ## Getting to business: Training and comparing closure models
#
# If you have made it this far, you will probably have noticed that we have
# not yet done any computation (apart from one anecdotical Burgers simulation).
# We have only defined functions: losses, initializers, models, training...
#
# Now we set up an experiment. We need to decide on the following:
#
# - Problem parameter: $\nu$
# - LES resolution
# - DNS resolution
# - Discrete filter
# - Number of initial conditions
# - Closure model: FNO and CNN
# - Simulation time: Too short, and we won't have time to detect instabilities
#   created by our model; too long, and most of the data will be too smooth for
#   a closure model to be needed (due to viscosity)
#
# In addition, we will split our data into
#
# - Training data (for choosing $\theta$)
# - Validation data (just for monitoring training, choose when to stop)
# - Testing data (for testing performance on unseen data)

# ### Discretization and filter
#
# We will use a Gaussian filter kernel, truncated to zero outside of $3 / 2$
# filter widths.

## Resolution
nx_les = 64
nx_dns = 512

## Grids
x_les = LinRange(0.0, 1.0, nx_les + 1)[2:end]
x_dns = LinRange(0.0, 1.0, nx_dns + 1)[2:end]

## Grid sizes
Δx_les = 1 / nx_les
Δx_dns = 1 / nx_dns

## Filter width
Δϕ = 3 * Δx_les

## Filter kernel
gaussian(Δ, x) = sqrt(6 / π) / Δ * exp(-6x^2 / Δ^2)
top_hat(Δ, x) = (abs(x) ≤ Δ / 2) / Δ
kernel = gaussian

## Discrete filter matrix (with periodic extension and threshold for sparsity)
ϕ = sum(-1:1) do z
    d = @. x_les - x_dns' - z
    @. kernel(Δϕ, d) * (abs(d) ≤ 3 / 2 * Δϕ)
end
ϕ = ϕ ./ sum(ϕ; dims = 2) ## Normalize weights
ϕ = sparse(ϕ)
dropzeros!(ϕ)
heatmap(ϕ; yflip = true, xmirror = true, title = "Filter matrix")

# ### Create data
#
# Create the training, validation, and testing datasets. 
# Use a different time step for testing to detect overfitting.

ν = 1.0e-3
data_train = create_data(10, ϕ; nt = 1000, dt = 1.0e-3, ν)
data_valid = create_data(2, ϕ; nt = 100, dt = 1.1e-3, ν)
data_test = create_data(5, ϕ; nt = 1000, dt = 0.8e-3, ν)

# ### Closure models
#
# We also include a "no closure" model (baseline for comparison).

noclosure, θ_noclosure = (u, θ) -> zero(u), nothing

# #### Train a CNN
#
# Create CNN model. Note that the last activation is `identity`, as we don't
# want to restrict the output values. We can inspect the structure in the
# wrapped Lux `Chain`.

cnn, θ_cnn = create_cnn(;
    r = [2, 2, 2, 2],
    channels = [8, 8, 8, 1],
    σ = [leakyrelu, leakyrelu, leakyrelu, identity],
    use_bias = [true, true, true, false],
    rng,
)
cnn.chain

# Choose loss function

loss = create_randloss_commutator(cnn, data_train; nuse = 50)
## loss = create_randloss_trajectory(g, data_train; nuse = 3, nunroll = 10, ν, m = cnn)

# Initilize CNN training state

trainstate = initial_trainstate(Adam(1.0e-3), θ_cnn)

# Model warm-up: trigger compilation and get indication of complexity

loss(θ_cnn)
gradient(loss, θ_cnn);
@time loss(θ_cnn);
@time gradient(loss, θ_cnn);

# Train the CNN. The cell below can be repeated to continue training where the
# previous training session left off.

trainstate = train(;
    trainstate...,
    loss,
    niter = 2000,
    ncallback = 20,
    callback = create_callback(cnn, data_valid),
)

# Final CNN weights

θ_cnn = trainstate.θ

# #### Train an FNO
#
# Create FNO. Like for the CNN, last activation is `identity`.

fno, θ_fno = create_fno(;
    channels = [5, 5, 5, 5],
    kmax = [16, 16, 16, 8],
    σ = [gelu, gelu, gelu, identity],
    rng,
)
fno.chain

# Choose loss function

loss = create_randloss_commutator(fno, data_train; nuse = 50)
## loss = create_randloss_trajectory(g, data_train; nuse = 3, nunroll = 10, ν, m = fno)

trainstate = initial_trainstate(Adam(1.0e-3), θ_fno)

## Model warm-up: trigger compilation and get indication of complexity
loss(θ_fno);
gradient(loss, θ_fno);
@time loss(θ_fno);
@time gradient(loss, θ_fno);

# Train the FNO. The cell below can be repeated to continue training where the
# previous training session left off.

trainstate = train(;
    trainstate...,
    loss,
    niter = 1000,
    ncallback = 20,
    callback = create_callback(fno, data_valid),
)

# Final FNO weights
θ_fno = trainstate.θ

# ### Model performance
#
# We will now make a comparison of the three closure models (including the
# "no-model" where $m = 0$, which corresponds to solving the DNS equations on the
# LES grid).

m = [noclosure, cnn, fno]
θ = [θ_noclosure, θ_cnn, θ_fno]
labels = ["0", "CNN", "FNO"]

println("Relative a posteriori errors:")
for i in eachindex(m)
    e = trajectory_error(g, data_test.u; data_test.dt, ν, m = m[i], θ = θ[i])
    print(labels[i])
    print(":\t ")
    print(e)
    println()
end

# ## Modeling exercices
#
# To get confident with modeling ODE right hand sides using machine learning,
# it can be useful to do one or more of the following exercices.
#
# ### Trajectory fitting (a posteriori loss function)
#
# - Fit a closure model using the a posteriori loss function.
# - Investigate the effect of the parameter `nunroll`. Try for example `@time
#   randloss(θ)` for `unroll = 10` and `nunroll = 20` (execute `randloss` once
#   first to trigger compilation).
# - Discuss the statement "$L^\text{prior}$ and $L^\text{post}$ are almost the
#   same when `nunroll = 1`" with your neighbour. Are they exactly the same if
#   we use forward Euler ($u^{n + 1} = u^n + \Delta t f(u^n)$) instead of RK4?
#
# ### Neural ODE (brute force right hand side)
#
# 1. Observe that, if we really want to, we can skip the term $f(\bar{u})$
#    entirely, hoping that $m$ will be able to model it directly (in addition
#    to the commutator error). The resulting model is
#    $$
#    \frac{\mathrm{d} \bar{v}}{\mathrm{d} t} = m(\bar{v}, \theta).
#    $$
#    This is known as a _Neural ODE_ [^5].
# 1. Rewrite the function `g` such that the closure model predicts the _entire_
#    right hand side instead of the correction only. (Comment out `f` in `g`).
# 1. Train the CNN or FNO in this setting. Is the model able to represent the
#    solution correctly?
#
# ### Learn the discretization
#
# - Make a new instance of the CNN closure, called `cnn_linear` with parameters
#   `θ_cnn_linear`, which only has one convolutional layer.
#   This model should still add the square input channel.
# - Observe that the original Burgers DNS RHS $f$ can actually be expressed in
#   its entirety using this model, i.e.
#   $$
#   \frac{\mathrm{d} u}{\mathrm{d} t} = f(u) = \operatorname{CNN}(u,
#   \theta).
#   $$
#   - What is the kernel radius?
#   - Should there still be a nonlinear activation function?
#   - What is the exact expression for the model weights and bias?
# - "Improve" the discretization $f$: Increase the kernel radius of
#   `cnn_linear` and train the model. What does the resulting kernel stencil
#   (`Array(θ_cnn_linear)`) look like? Does it resemble the one of $f$?
#
# ### Naive neural closure model
#
# Create a new instance of the CNN or FNO models, called `cnn_naive` or
# `fno_naive`, without the additional square input channel (prior physical
# knowledge). The input should only have one singleton channel (pass the
# keyword argument `input_channels = (u -> u,)` to the constructor).
# Do you expect this version to perform better or worse than with a square
# channel?
#
# ## References
#
# [^3]: Z. Li, N. Kovachki, K. Azizzadenesheli, B. Liu, K. Bhattacharya, A.
#       Stuart, and A. Anandkumar.
#       _Fourier neural operator for parametric partial differential
#       equations._
#       arXiv:[2010.08895](https://arxiv.org/abs/2010.08895), 2021.
#
# [^4]: Hugo Melchers, Daan Crommelin, Barry Koren, Vlado Menkovski, Benjamin
#       Sanderse,
#       _Comparison of neural closure models for discretised PDEs_,
#       Computers & Mathematics with Applications,
#       Volume 143,
#       2023,
#       Pages 94-107,
#       ISSN 0898-1221,
#       <https://doi.org/10.1016/j.camwa.2023.04.030>.
#
# [^5]: R. T. Q. Chen, Y. Rubanova, J. Bettencourt, and D. Duvenaud.
#       _Neural Ordinary Differential Equations_.
#       arXiv:[1806.07366](https://arxiv.org/abs/1806.07366), 2018.
