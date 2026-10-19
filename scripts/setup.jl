using Pkg
# Add custom registry for the RDE.jl and RDE_Env.jl packages
Pkg.status()
Pkg.Registry.add(url = "https://github.com/KristianHolme/KristianHolmeRegistry.git")
#activate base julia environment and add DrWatson
Pkg.activate()
Pkg.add("DrWatson")

# Activate DRL_RDE project root (independent of current working directory)
project_dir = dirname(@__DIR__)
@assert isfile(joinpath(project_dir, "Project.toml")) "Expected Project.toml in repo root: $(repr(project_dir))"
Pkg.activate(project_dir)

# instantiate project with dependencies
@info "Instantiating project with dependencies. Precompilation can take some time..."
Pkg.instantiate()
@info "Project instantiated successfully."
