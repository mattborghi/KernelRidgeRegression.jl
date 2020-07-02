# utility functions

function make_blocks(nobs, nblocks)
    maxbs, reminder = divrem(nobs, nblocks)

    res = fill(maxbs, nblocks)
    if reminder > 0
        res[1:reminder] .= maxbs + 1
    end
    res
end

# the truncated newton method for matrix inversion
# adapted from https://en.wikipedia.org/wiki/Conjugate_gradient_method
# solves Ax = b for x, overwrites x
# stops if the error is < ɛ or after reaching max_iter
function truncated_newton!(A::Matrix{T}, b::Vector{T},
                              x::Vector{T}, ɛ::T, max_iter::Int) where T
    r = b - A * x
    p = deepcopy(r)
    Ap = deepcopy(r)
    rsold = dot(r, r)

    n = length(r)

    for i in 1:max_iter
        # Ap[:] = A * p
        mul!(Ap, A, p)
        α = rsold / dot(p, Ap)
        # x += α * p
        BLAS.axpy!(α, p, x)
        # r -= α * Ap
        BLAS.axpy!(-α, Ap, r)
        rsnew = dot(r, r)
        rsnew < ɛ &&  break
        β = rsnew / rsold
        # p[:] = r + β * p
        BLAS.scal!(n, β, p, 1)
        BLAS.axpy!(1, r, p)
        rsold = rsnew
    end
    return x
end

function unmerge_values(data::Vector{T}, assignments::Vector{Int}, n_clusters::Int) where {T <: AbstractFloat}
    ŷ = similar(data)
    elems = map(i -> sum(assignments .== i), 1:n_clusters)
    Idxs = ones(Int64, n_clusters)
    @views @inbounds for i in 1:n_clusters - 1
      # Loop on previous 
        Idxs[i + 1] = sum(elems[1:i]) + 1
    end
    @views @inbounds for i in 1:length(data)
        current_cluster = assignments[i]
        search_index = Idxs[current_cluster]
        
        ŷ[i] = data[search_index]
      # Update corresponding cluster 
        Idxs[current_cluster] += 1  
    end
    return ŷ
end