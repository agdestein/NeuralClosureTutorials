using Literate
using FileWatching

filename = "navier_stokes_spectral.jl"
filename = "burgers.jl"

Literate.notebook(filename; execute = false)

while true
    # Regenerate markdown file at every change
    watch_file(filename)
    Literate.markdown(filename; codefence = "```julia" => "```")
end
