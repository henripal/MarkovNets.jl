module MarkovNets

using Compat
using Reexport

@reexport using ProbabilisticGraphicalModels
import TikzGraphs: plot
import LightGraphs: Graph, add_edge!, neighbors,
                    has_edge, connected_components,
                    vertices
                    
import Factors

export 
    MarkovNet,
    UG,
    MNGibbsSampler,

    neighbors,
    has_edge,
    markov_blanket,
    is_independent,
    evidence_reduce


include("markov_nets.jl")
include("mn_gibbs.jl")

end # module
