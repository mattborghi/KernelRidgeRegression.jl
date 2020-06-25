using KernelRidgeRegression
using Plots
using BenchmarkTools
# using Gadfly
# @everywhere using ParallelDataTransfer

N = 5000
x = rand(1, N) * 4π .- 2π
yy = sinc.(x) # vec(sinc.(4 .* x) .+ 0.2 .* sin.(30 .* x))
y = dropdims(yy + 0.1 * randn(1, N), dims=1)
# xnew = collect(-2.5π:0.01:2.5π)'
xnew = reshape(collect(-2.5π:0.01:2.5π), 1, :)
# xnew = reshape(collect(-2.5π:π / 1000:2.5π), 1, :)
# sendto(workers(), x = x, y = y)

println("KRR")
# BLAS.set_num_threads(2)
mykrr = KernelRidgeRegression.fit(KernelRidgeRegression.KRR, x, y, 1e-3 / 5000, KernelFunctions.GaussianKernel())
ynew = KernelRidgeRegression.predict(mykrr, xnew)
@btime mykrr = KernelRidgeRegression.fit(KernelRidgeRegression.KRR, x, y, 1e-3 / 5000, KernelFunctions.GaussianKernel())
@btime ynew = KernelRidgeRegression.predict(mykrr, xnew)

println("fastKRR")
myfastkrr = KernelRidgeRegression.fit(KernelRidgeRegression.FastKRR, x, y, 4 / 5000, 10, KernelFunctions.GaussianKernel())
yfastnew = KernelRidgeRegression.predict(myfastkrr, xnew)
@btime myfastkrr = KernelRidgeRegression.fit(KernelRidgeRegression.FastKRR, x, y, 4 / 5000, 10, KernelFunctions.GaussianKernel())
@btime yfastnew = KernelRidgeRegression.predict(myfastkrr, xnew)

println("ParFastKRR")
# BLAS.set_num_threads(1)
myparfastkrr = KernelRidgeRegression.fitPar(
    KernelRidgeRegression.FastKRR,
    N,
    i->x[:,i], i->y[i],
    4 / 5000, 10,
    KernelFunctions.GaussianKernel()
)
yparfastnew = KernelRidgeRegression.predict(myfastkrr, xnew)
@btime myparfastkrr = KernelRidgeRegression.fitPar(
    KernelRidgeRegression.FastKRR, N,
    i->x[:,i], i->y[i],
    4 / 5000, 10,
    KernelFunctions.GaussianKernel()
)
@btime yparfastnew = KernelRidgeRegression.predict(myfastkrr, xnew)

# draw(SVG("test.svg", 20cm, 20cm), plot(
#     layer(x=xnew, y=yparfastnew, Geom.line, Theme(default_color=colorant"red")),
#     layer(x=xnew, y=yfastnew,    Geom.line, Theme(default_color=colorant"purple")),
#     layer(x=xnew, y=ynew,        Geom.line, Theme(default_color=colorant"green")),
#     layer(x=x,    y=y,           Geom.line, Theme(default_color=colorant"blue")),
#     Coord.cartesian(ymin=-1.5, ymax=1.5),
#     Guide.manual_color_key(
#         "Method",
#         ["Data",         "KRR",           "FastKRR",        "ParFastKRR" ],
#         [colorant"blue", colorant"green", colorant"purple", colorant"red"]
#     )
# ))

scatter(x', y) 
scatter!(xnew', ynew)
scatter!(xnew', yfastnew)
scatter!(xnew', yparfastnew)