
__precompile__()
module KernelRidgeRegression

export KRR,
    FastKRR,
    RandomFourierFeatures,
    TruncatedNewtonKRR,
    NystromKRR,
    fitPar

import Base: show, showcompact, display
import MLKernels
import StatsBase
import StatsBase: fit, fitted, predict, nobs, predict!, RegressionModel, sample

abstract AbstractKRR{T} <: RegressionModel

function fit(::Type{AbstractKRR}) error("not implemented") end

StatsBase.fitted(KRR::AbstractKRR) = predict(KRR, KRR.X)

include("krr.jl")
include("util.jl")

end # module KernelRidgeRegression
