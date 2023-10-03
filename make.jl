using Literate

files = ["burgers.jl", "navier_stokes_spectral.jl"]
output_dir = "generated"

ispath(output_dir) || mkpath(output_dir)

for f in files
    Literate.notebook(f, output_dir; execute = false)
    Literate.markdown(f, output_dir; codefence = "```julia" => "```")
end
