
abstract MarkovNetSampler

"""
The MarkovNetGibbsSampler type houses the parameters of the Gibbs sampling algorithm.  The parameters are defined below:

burn_in:  The first burn_in samples will be discarded.  They will not be returned.
The thinning parameter does not affect the burn in period.
This is used to ensure that the Gibbs sampler converges to the target stationary distribution before actual samples are drawn.

thinning: For every thinning + 1 number of samples drawn, only the last is kept.
Thinning is used to reduce autocorrelation between samples.
Thinning is not used during the burn in period.
e.g. If thinning is 1, samples will be drawn in groups of two and only the second sample will be in the output.

evidence: the assignment that all samples must be consistent with (ie, Assignment(:A=>1) means all samples must have :A=1).
Use to sample conditional distributions.

initial_sample:  The inital assignment to variables to use.  If null, the initial sample is chosen at random
"""
type MNGibbsSampler <: MarkovNetSampler

    evidence::Assignment
    burn_in::Int
    thinning::Int
    initial_sample::Nullable{Assignment}

    function MNGibbsSampler(evidence::Assignment=Assignment();
                             burn_in::Int=100,
                             thinning::Int=0,
                             initial_sample::Nullable{Assignment}=Nullable{Assignment}()
                             )
        new(evidence, burn_in, thinning, initial_sample)
    end
end
    
"""
Implements Gibbs sampling for MarkovNets.
This Gibbs sample only supports discrete MarkovNets, and samples are
drawn following a Categorical Distribution with probabilities
equal to the normalized potentials

Sampling requires an MarkovNetGibbsSampler object which contains the parameters
"""
function Base.rand(mn::MarkovNet, sampler::MNGibbsSampler, nsamples::Integer)

    return gibbs_sample(mn, nsamples, sampler.burn_in, thinning=sampler.thinning, evidence=sampler.evidence, initial_sample=sampler.initial_sample)
end


function gibbs_sample(mn::MarkovNet, nsamples::Integer, burn_in::Integer;
                      thinning::Integer=0,
                      evidence::Assignment=Assignment(),
                      initial_sample::Nullable{Assignment}=Nullable{Assignment}()
                      )
    # Check parameters for correctness
    nsamples > 0 || throw(ArgumentError("nsamples parameter less than 1"))
    burn_in >= 0 || throw(ArgumentError("Negative burn_in parameter"))
    if ~ isnull(initial_sample)
        init_sample = get(initial_sample)
        for vertex in vertices(mn.ug)
            haskey(init_sample, Symbol(vertex)) || throw(ArgumentError("Gibbs sample initial_sample must be an assignment with all variables in the Bayes Net"))
        end
        for (vertex, value) in evidence
            init_sample[vertex] == value || throw(ArgumentError("Gibbs sample initial_sample was inconsistent with evidence"))
        end
    end

    # reduce the MarkovNet according to the evidence:
    mn_reduced = evidence_reduce(mn, evidence)

    if isnull(initial_sample)
        # Hacky
        initial_sample = Assignment()
        for factor in mn_reduced.factors
            for (i, vertex) in enumerate(factor.dimensions)
                n_categories = length(factor.potential[:, i])
                initial_sample[vertex.name] = rand(Categorical(n_categories))
            end
        end
    end


    # create the data frame
    t = Dict{Symbol, Vector{Any}}()
    for name in keys(initial_sample)
        t[name] = Any[]
    end
    
    # initialize the sample to our initial sample
    current_sample = initial_sample

    # burn in, if present
    if burn_in != 0
        for burn_in_sample in 1:burn_in
            current_sample = gibbs_sample_loop(mn_reduced, current_sample)
        end
    end

    # main loop
    for sample_iter in 1:nsamples

        # first skip over the thinning 
        for skip_iter in 1:thinning
            current_sample = gibbs_sample_loop(mn_reduced, current_sample)
        end
        
        # real loop, we store the values in the dict
        current_sample = gibbs_sample_loop(mn_reduced, current_sample, t)
           
    end

    return DataFrame(t)
end

        
function gibbs_sample_loop(mn::MarkovNet, current_sample::Assignment,
                           results::Dict{Symbol, Vector{Any}} = Dict{Symbol, Vector{Any}}())::Assignment
    sample::Assignment = Assignment()
    other_assg::Assignment = Assignment()
    
    for v in vertices(mn.ug)
            
        other_assg = copy(current_sample) # this is faster than filter
        delete!(other_assg, Symbol(v))
        new_f = reduce(*, [mn.factors[factor_index][other_assg] for factor_index in mn.name_to_factor_indices[Symbol(v)]])
        proba = new_f.potential[:]
        proba = proba/sum(proba)
        sample_index = rand(Categorical(proba))
        current_sample[Symbol(v)] = sample_index
        if ~ isempty(results)
            push!(results[Symbol(v)], current_sample[Symbol(v)])
        end
    end
    return current_sample
end


