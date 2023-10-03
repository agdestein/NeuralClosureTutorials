using Literate
using FileWatching

files = ["burgers.jl", "navier_stokes_spectral.jl"]
output_dir = "generated"

ispath(output_dir) || mkpath(output_dir)

for f in files
    Literate.notebook(f, output_dir; execute = false)
    Literate.markdown(f, output_dir; codefence = "```julia" => "```")
end

# filename = "navier_stokes_spectral.jl"
# filename = "burgers.jl"
#
# while true
#     # Regenerate markdown file at every change
#     watch_file(filename)
#     Literate.markdown(filename; codefence = "```julia" => "```")
# end
