using Plots
using MAT
using Serialization

t_mat = Vector{Float64}()
t_julad = Vector{Float64}()
t_julspm = Vector{Float64}()
t_julmtk = Vector{Float64}()
iter_spm = Vector{Float64}()
iter_ad = Vector{Float64}()
iter_mtk = Vector{Float64}()
F_spm = Vector{Float64}()
F_mat = Vector{Float64}()
F_ad =  Vector{Float64}()
F_mtk =  Vector{Float64}()
list = collect(2:12)
# append!(list,10)
for i = list
    tmp = matread("speed-comparison/speedcomp_fmri_" * string(i) * "regions.mat");
    tmpmat = matread("speed-comparison/fmri_" * string(i) * "regions.mat");
    # append!(t_mat, tmp["t_mat"]/tmp["iter_spm"])
    # append!(t_julad, tmp["t_jad"]/tmp["iter_ad"])
    # append!(t_julspm, tmp["t_jspm"]/tmp["iter_spm"])
    # append!(t_mat, tmp["t_mat"])
    append!(t_mat, tmpmat["matcomptime"])
    append!(t_julad, tmp["t_jad"])
    append!(t_julspm, tmp["t_jspm"])
    append!(t_julmtk, tmp["t_mtk"])
    append!(iter_ad, tmp["iter_ad"])
    append!(iter_spm, tmp["iter_spm"])
    append!(iter_mtk, tmp["iter_mtk"])
    append!(F_mat, tmp["F_mat"])
    append!(F_spm, tmp["F_jspm"])
    append!(F_ad, tmp["F_jad"])
    append!(F_mtk, tmp["F_mtk"])
end
# plot(list, (t_mat./t_julad.-1), legendfontsize=12, xtickfontsize=14, xguidefontsize=14, ytickfontsize=14, yguidefontsize=14, legend=:topleft, xlabel="number of regions",ylabel="speed increase", linewidth=3, label="Julia AD")
# plot!(list, (t_mat./t_julspm.-1), linewidth=3, label="Julia SPM12")
# scatter!(list, (t_mat./t_julad.-1), markersize=6, color="blue", label=nothing)
# scatter!(list, (t_mat./t_julspm.-1), markersize=6, color="red", label=nothing)
plot(list, t_julspm./60, titlefontsize=25, legendfontsize=25, xticks=list, xtickfontsize=25, xguidefontsize=25, ytickfontsize=25, yguidefontsize=25, legend=:topleft, xlabel="Number of Regions",ylabel="Computation Time [min]", linewidth=5, label="Julia SPM25", framestyle=:box, left_margin=20px, bottom_margin = 22px, size=(800,500))
plot!(list, t_julad./60, linewidth=5, label="Julia AD")
plot!(list, t_julmtk./60, linewidth=5, label="Julia MTK + AD")
plot!(list, t_mat./60, linewidth=5, color="black", label="Matlab SPM25")
# scatter!(list, t_julad, markersize=6, color="blue", label=nothing)
# scatter!(list, t_julspm, markersize=6, color="red", label=nothing)
title!("fMRI - linear neural mass models")
savefig("speed-comparison/plots/Fig 3a - speedcomp12regions_fMRI.png")


############## LFP/CMC Speed Comparison ####################

t_mat = Vector{Float64}()
t_julmtk = Vector{Float64}()
iter_mtk = Vector{Float64}()
F_mat = Vector{Float64}()
F_mtk =  Vector{Float64}()
list = collect(2:10)
for i = list
    tmp = matread("speed-comparison/speedcomp_lfp_" * string(i) * "regions.mat");
    # tmpmat = matread("speed-comparison/cmc_" * string(i) * "regions.mat");
    # append!(t_mat, tmp["t_mat"]/tmp["iter_spm"])
    # append!(t_julad, tmp["t_jad"]/tmp["iter_ad"])
    # append!(t_julspm, tmp["t_jspm"]/tmp["iter_spm"])
    # append!(t_mat, tmp["t_mat"])
    append!(t_mat, tmp["t_mat"])
    append!(t_julmtk, tmp["t_mtk"])
    append!(iter_mtk, tmp["iter_mtk"])
    append!(F_mat, tmp["F_mat"])
    append!(F_mtk, tmp["F_mtk"])
end
# plot(list, (t_mat./t_julad.-1), legendfontsize=12, xtickfontsize=14, xguidefontsize=14, ytickfontsize=14, yguidefontsize=14, legend=:topleft, xlabel="number of regions",ylabel="speed increase", linewidth=3, label="Julia AD")
# plot!(list, (t_mat./t_julspm.-1), linewidth=3, label="Julia SPM12")
# scatter!(list, (t_mat./t_julad.-1), markersize=6, color="blue", label=nothing)
# scatter!(list, (t_mat./t_julspm.-1), markersize=6, color="red", label=nothing)
plot(list, t_julmtk./60, titlefontsize=25, legendfontsize=25, color=3, xticks=list, xtickfontsize=25, xguidefontsize=25, ytickfontsize=25, yguidefontsize=25, legend=:topleft, xlabel="Number of Regions", ylabel="Computation Time [min]", linewidth=5, label="Julia MTK + AD", framestyle=:box, left_margin=20px, bottom_margin = 22px, size=(800,500))
plot!(list, t_mat./60, linewidth=5, color="black", label="Matlab SPM25")
title!("LFP - canonical micro-ciruit")
savefig("speed-comparison/plots/Fig 3b - speedcomp10regions_LFP.png")




### plot ADVI results ###
using JLD2
using StatsPlots
using LinearAlgebra
using LaTeXStrings
using MAT

r = 3
vals = matread("speedandaccuracy/matlab_" * string(r) * "regions.mat");
A_true = vals["true_params"]["A"]
nd = size(A_true, 1)

tp = Dict(
    :A => vals["true_params"]["A"],
    :lntransit => vals["true_params"]["transit"],
    :lndecay => 0.0,
    :lnϵ => 0.0,
    :α => zeros(2),
    :β => zeros(2),
    :γ => zeros(3),
    :csd => vals["csd"]
)

vlp = Dict(  # variational Bayes results
    :A => vals["Ep"]["A"],
    :lntransit => vals["Ep"]["transit"],
    :lndecay => vals["Ep"]["decay"][1],
    :lnϵ => vals["Ep"]["epsilon"][1],
    :α => vals["Ep"]["a"][1:2],
    :β => vals["Ep"]["b"][1:2],
    :γ => vals["Ep"]["c"][1:nd],
    :csd => vals[]
)

advip = Dict( # ADVI results
    :A => [],
    :lntransit => [],
    :lndecay => [],
    :lnϵ => [],
    :α => [],
    :β => [],
    :γ => [],
    :csd => []
)


ADVIsteps = 1000
ADVIsamples = 10

for iter = 1:4
    q = load_object("speedandaccuracy/newADVI/ADVI" * string(iter) * "_sa" * string(ADVIsamples) * "_st" * string(ADVIsteps) * "_r" * string(r) * ".jld2")[1];
    push!(advip[:A], reshape(q.dist.m[q.transform.ranges[1]], d, d))
    push!(advip[:lntransit], q.dist.m[q.transform.ranges[2]])
    push!(advip[:lndecay], q.dist.m[q.transform.ranges[3]])
    push!(advip[:lnϵ], q.dist.m[q.transform.ranges[4]])
    push!(advip[:α], q.dist.m[q.transform.ranges[5]])
    push!(advip[:β], q.dist.m[q.transform.ranges[6]])
    push!(advip[:γ], q.dist.m[q.transform.ranges[7]])
    push!(advip[:csd], q.dist.m[q.transform.ranges[8]] + im*q.dist.m[q.transform.ranges[9]])
end


push!(advip[:A], mean(advip[:A]))
push!(advip[:lntransit], mean(advip[:lntransit]))
push!(advip[:lndecay], mean(advip[:lndecay]))
push!(advip[:lnϵ], mean(advip[:lnϵ]))
push!(advip[:α], mean(advip[:α]))
push!(advip[:β], mean(advip[:β]))
push!(advip[:γ], mean(advip[:γ]))

abs.((advip[:A][end] - tp[:A]))
abs.(vlp[:A] - tp[:A])
abs.(advip[:lntransit][end] - tp[:lntransit])
abs.(vlp[:lntransit] - tp[:lntransit])

abs.(advip[:lndecay][end] - tp[:lndecay])
abs.(vlp[:lndecay] - tp[:lndecay])


A_true = vals["true_params"]["A"]
d = size(A_true, 1)
A_std = reshape(sqrt.(diag(vals["Cp"][1:d^2,1:d^2])), d, d)
ADVIsteps = 1000
ADVIsamples = 10
iter = 1
(q, advi, cond_model) = load_object("speedandaccuracy/fixedADVI/ADVI" * string(iter) * "_sa" * string(ADVIsamples) * "_st" * string(ADVIsteps) * "_r" * string(r) * ".jld2");
# (q, advi, cond_model) = deserialize("speedandaccuracy/ADVIADA" * string(iter) * "_sa" * string(ADVIsamples) * "_st" * string(ADVIsteps) * "_0.01_r" * string(r) * ".dat");
Turing.elbo(advi, q, cond_model, 10)
Fs = zeros(100)
for iter in 1:15
    (q, advi, cond_model) = deserialize("speedandaccuracy/ADVIADA" * string(iter) * "_sa" * string(ADVIsamples) * "_st" * string(ADVIsteps) * "_0.01_r" * string(r) * ".dat");
    for i = 1:100
        Fs[i] = Turing.elbo(advi, q, cond_model, 10)
    end
    histogram(Fs, xlims=(-5000, 0))
    savefig("speedandaccuracy/plots/F_ADVI_hist" * string(iter) * "_sa" * string(ADVIsamples) * "_st" * string(ADVIsteps) * "_0.01_r" * string(r) * ".png")
end


nuts = load_object("speedandaccuracy/NUTS/MCMC_NUTS_sa1000.jld2")
nuts = load_object("speedandaccuracy/NUTS/MCMC_NUTS_std10_sa1000.jld2")
nuts.value[end, :, 1]
ss = describe(nuts)[1]
nutsm = reshape(ss.nt.mean[1:d^2], d, d)
nutss = reshape(ss.nt.std[1:d^2], d, d)

X = []
Yj = []
Yj_total = []
Ym = []
Ym_err = []
Yn = []
Yn_err = []
Yt = []
for i = 1:d
    for j = 1:d
        if i == j
            continue
        end
        push!(X, latexstring("a_{", string(i), string(j), "}"))
        push!(Yj, reshape(q.dist.m[1:d^2], d, d)[i,j])
        # push!(Yn, nutsm[i, j])
        # push!(Yn_err, nutss[i, j])
        push!(Ym, vals["Ep"]["A"][i, j])
        push!(Ym_err, A_std[i, j])
        push!(Yt, A_true[i, j])
    end
end
# push!(Yj_total, (Yj .- Yt)./abs.(Yt))
push!(Yj_total, Yj)
# scatter(X, Yj, label="ADVI",color=:orange, legendfontsize=10, xtickfontsize=12, xguidefontsize=12, ytickfontsize=12, yguidefontsize=12, markeralpha=0.3)
for iter = 2:6
    q = load_object("speedandaccuracy/fixedADVI/ADVI" * string(iter) * "_sa" * string(ADVIsamples) * "_st" * string(ADVIsteps) * "_r" * string(r) * ".jld2")[1];
    Yj = []
    for i = 1:d
        for j = 1:d
            if i == j
                continue
            end
            push!(Yj, reshape(q.dist.m[1:d^2], d, d)[i,j])
        end
    end
    # scatter!(X, Yj, label=false, color=:orange, markeralpha=0.3)
    # push!(Yj_total, (Yj .- Yt)./abs.(Yt))
    push!(Yj_total, Yj)
end

default(size=(800,600),
        legendfontsize=14,
        xtickfontsize=22,
        xguidefontsize=16,
        ytickfontsize=16,
        yguidefontsize=16,
        markersize=8,
        # ylabel="relative distance from ground truth")
        ylabel="connection strength")

boxplot(repeat(X,outer=5), vcat(Yj_total...), label="ADVI")
# scatter!(X, (Ym .- Yt)./abs.(Yt), yerr=Ym_err./abs.(Yt), label="Laplace", legend=:best)
scatter!(X, Ym, yerr=Ym_err, label="Laplace", legend=:best)
# scatter!(X, (Yn .- Yt)./abs.(Yt), yerr=Yn_err./abs.(Yt), label="NUTS")
scatter!(X, Yn, yerr=Yn_err, label="NUTS")
scatter!(X, Yt, label="True", markershape=:diamond)
hline!([0], linestyle=:dash, color=:black, label=false)

savefig("speedandaccuracy/plots/newADVIboxplot_NUTSstd_sa" * string(ADVIsamples) * "_st" * string(ADVIsteps) * "_0.01_r" * string(r) * ".png")

savefig("speedandaccuracy/plots/boxplot_NUTS1000_std1_r" * string(r) * ".png")
scatter!(X, Ym  .- Yt, label="Laplace", color=:blue)
scatter!(repeat(X,outer=5),Yj_total, label=false, markeralpha=0.3, color=:orange)
scatter!(X, mean(Yj_total), yerr=std(Yj_total), label=L"$\mu$(ADVI)", color=:green, markerhsape=:diamond)
title!("standard deviation of interaction = 0.01\n ADVI samples = " * string(ADVIsamples) * ", steps =" * string(ADVIsteps) * ", regions = " * string(r))
violin(repeat(X,outer=5), vcat(Yj_total...), linewidth=0)
scatter!(X, Ym  .- Yt, label="Laplace", color=:blue)
boxplot!(repeat(X,outer=5), vcat(Yj_total...), fillalpha=0.75, linewidth=2)


using JLD2

