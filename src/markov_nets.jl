#=
A Markov Random Field  (MarkovNet) represents a probability distribution
over a set of variables, P(x₁, x₂, ..., xₙ)
It leverages relations between variables in order to efficiently encode the joint distribution.
A MarkovNet is defined by an undirected graph in which each node is a variable
and contains an associated conditional probability distribution P(xⱼ | parents(xⱼ))
=#
typealias UG Graph
function _build_ug_from_factors{T<:Factors.Factor}(
    factors::AbstractVector{T},
    name_to_index::Dict{NodeName, Int},
    name_to_factor_indices::Dict{NodeName, Array{Int}}
    )
    
    # Dict already has unique node names
    ug = UG(length(name_to_index))

    # Build the UG by linking all edges within a given factor
    for (factor_index, factor) in enumerate(factors)
        for d in factor.dimensions
            if ~ (factor_index in name_to_factor_indices[d.name])
                push!(name_to_factor_indices[d.name], factor_index)
            end
        end
        
        for d1 in factor.dimensions, d2 in factor.dimensions        
            i, j = name_to_index[d1.name], name_to_index[d2.name]
            if i <j
                add_edge!(ug, i, j)
            end
        end
    end

    ug
end

type MarkovNet{T<:Factors.Factor}
    ug::UG 
    factors::Vector{T} # the factors associated with the MarkovNet
    names::Vector{NodeName}
    name_to_index::Dict{NodeName,Int} # NodeName → index in ug 
    name_to_factor_indices::Dict{NodeName, Array{Int}}
end
MarkovNet() = MarkovNet(UG(0), Factors.Factor[], NodeName[], Dict{NodeName, Int}(), Dict{NodeName, Array{Int}}())
MarkovNet{T <: Factors.Factor}(::Type{T}) = mn(UG(0), T[], NodeName[], Dict{NodeName, Int}(), Dict{NodeName, Array{Int}}())

function MarkovNet{T <: Factors.Factor}(factors::AbstractVector{T})
    name_to_index = Dict{NodeName, Int}()
    name_to_factor_indices = Dict{NodeName, Array{Int}}()
    names = Array{Symbol}[]
    # We need a collection of unique nodes to create the graph
    if isempty(names)
        names = unique(collect(Base.flatten([Factors.name.(factor.dimensions) for factor in factors])))
    end

    for (i, node) in enumerate(names)
        name_to_index[node] = i
        name_to_factor_indices[node] = []
    end

    ug = _build_ug_from_factors(factors, name_to_index, name_to_factor_indices)
    MarkovNet(ug, factors, names, name_to_index, name_to_factor_indices)

end

Base.get(mn::MarkovNet, i::Int) = mn.names[i]
Base.length(mn::MarkovNet) = length(mn.name_to_index)

"""
Returns the list of NodeNames
"""
function Base.names(mn::MarkovNet)
    retval = Array(NodeName, length(mn)) 
    for (key,val) in mn.name_to_index
        retval[val] = key
    end
    retval
end
    
"""
Returns the neighbors as a list of NodeNames
"""
function neighbors(mn::MarkovNet, target::NodeName)
    i = mn.name_to_index[target]
    NodeName[mn.names[j] for j in neighbors(mn.ug, i)]
end

"""
Returns the markov blanket - here same as neighbors
"""
function markov_blanket(mn::MarkovNet, target::NodeName)
    return neighbors(mn, target)
end

"""
Whether the MarkovNet contains the given edge
"""
function has_edge(mn::MarkovNet, source::NodeName, target::NodeName)::Bool
    u = get(mn.name_to_index, source, 0)
    v = get(mn.name_to_index, target, 0)
    u != 0 && v != 0 && has_edge(mn.ug, u, v)
end
   
"""
Returns whether the set of node names `x` is d-separated
from the set `y` given the set `given`
"""
function is_independent(mn::MarkovNet, x::AbstractVector{NodeName}, y::AbstractVector{NodeName}, given::AbstractVector{NodeName})
    ug_copy = copy(mn.ug)
    # we copy the mn, then remove all edges
    # from `given`; then calc the connected components
    # if x and y are in different connected components then they are independent

    x_index = [mn.name_to_index[node] for node in x]
    y_index = [mn.name_to_index[node] for node in y]
    g_index = [mn.name_to_index[node] for node in given]

    for g in g_index
        for n in neighbors(mn.ug, g)
            rem_edge!(ug_copy, g, n)
        end
    end

    conn_components = connected_components(ug_copy)

    for component in conn_components
        if !isempty(intersect(component, x_index))
            if !isempty(intersect(component, y_index))
                return false
            end
        end
    end

    return true
end

"""
Reduces the MarkovNet given evidence
"""
function evidence_reduce(mn::MarkovNet, evidence::Assignment)
    new_factors = Array{Factors.Factor,1}()

    for factor in mn.factors
        small_evidence = Assignment()
        for dimension in factor.dimensions
            try
                push!(small_evidence, dimension => evidence[dimension])
            end
        end
        push!(new_factors, factor[small_evidence])

    end

  MarkovNet(new_factors); 
    
end


#### IO to be reworked.


function plot(mn::MarkovNet)
    plot(mn.ug, AbstractString[string(s) for s in mn.names])
end


@compat function Base.show(f::IO, a::MIME"image/svg+xml", mn::MarkovNet)
    show(f, a, plot(mn))
end
