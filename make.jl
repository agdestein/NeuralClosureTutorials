using Literate
using FileWatching

Literate.notebook("tutorial.jl"; execute = false)

while true
    # Regenerate markdown file at every change
    watch_file("tutorial.jl")
    Literate.markdown("tutorial.jl"; codefence = "```julia" => "```")
end
