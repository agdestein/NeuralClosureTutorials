using Literate

function notebook_without_pythonruns(f; name = f, kwargs...)
    tempname = "_" * name
    Literate.notebook(f * ".jl"; name = tempname, kwargs...)
    @info "Removing Python cell indicators"
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

notebook_without_pythonruns("burgers"; name = "burgers_with_output", execute = true)
