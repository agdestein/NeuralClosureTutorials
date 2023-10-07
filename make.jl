using Literate

function notebook_without_pythonruns(f; name = f, kwargs...)
    tempname = "_" * name
    Literate.notebook(f * ".jl"; name = tempname, kwargs...)
    open(tempname * ".ipynb") do io
        open(name * ".ipynb"; write = true) do output
            for line in eachline(io)
                println(output, replace(line, "##PYTHONRUNTIME " => ""))
            end
        end
    end
    rm(tempname * ".ipynb")
    name * ".ipynb"
end

cd("tutorials")

files = ["burgers", "navier_stokes_spectral"]

for f in files
    # Literate.notebook(f * ".jl"; execute = false)
    notebook_without_pythonruns(f; execute = false)
    Literate.markdown(
        f * ".jl";
        flavor = Literate.CommonMarkFlavor(),
        codefence = "```julia" => "```",
    )
end
