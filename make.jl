using Literate

cd("tutorials")

files = ["burgers.jl", "navier_stokes_spectral.jl"]

for f in files
    Literate.notebook(f; execute = false)
    Literate.markdown(f; codefence = "```julia" => "```")
end
