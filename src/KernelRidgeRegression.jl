
__precompile__()
module KernelRidgeRegression

using Reexport
using Random
using Distributed
using LinearAlgebra
using Clustering

@reexport using KernelFunctions

export KRR,
    FastKRR,
    RandomFourierFeatures,
    TruncatedNewtonKRR,
    NystromKRR,
    SubsetRegressorsKRR,
    fitPar,
    fit, predict, fit_and_predict

import Base: show, display # showcompact
import StatsBase
import StatsBase: fit, fitted, predict, nobs, predict!, RegressionModel, sample, weights

abstract type AbstractKRR{T} <: RegressionModel end

function fit(::Type{AbstractKRR}) error("not implemented") end

StatsBase.fitted(KRR::AbstractKRR) = predict(KRR, KRR.X)

include("krr.jl")
include("util.jl")

end # module KernelRidgeRegression
