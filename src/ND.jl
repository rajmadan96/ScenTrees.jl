using JuMP, LinearAlgebra, OptimalTransport, Tulip
include("TreeStructure.jl")
include("TreeApprox.jl")
include("StochPaths.jl")
include("LatticeApprox.jl")
include("KernelDensityEstimation.jl")
include("bushinessNesDistance.jl")
#include("Wasserstein.jl")

function distFunction(states1::Vector{Int64}, states2::Vector{Int64})::Array{Int64,2}
    n1= length(states1); n2= length(states2)
    dMatrix= Array{Int64}(undef, (n1,n2))
    for i= 1:n1, j= 1:n2
		dMatrix[i,j]= abs(states2[j]- states1[i])
    end
    return dMatrix
end



function nested_distance(tree1::Tree, tree2::Tree, r = 1)
    """
        Computes the Nested Distance between two given trees of same height for parameter r = 1;
    """
    T = height(tree1)+1; # T stage (the root is at height 0)
    D = zeros(Float64, (length(nodes(tree1)), length(nodes(tree2)))); # Alot will remain 0
    # Initialization, computes distance at time T 
    D[1,1] = (abs(tree1.state[1] - tree2.state[1]))^2;
    for t in 2:T
        nodes1 = nodes(tree1, t-1); # all nodes at stage t of tree1
        nodes2 = nodes(tree2, t-1);
        for i in nodes1
           for j in nodes2
                D[i,j] = D[tree1.parent[i], tree2.parent[j]] + (abs(tree1.state[i] - tree2.state[j]))^2;
            end
        end
    end
    D.^=(1/2);
    # Recursive, backward in time
    for t in T-1:-1:1 
        nodes1 = nodes(tree1, t-1); 
        nodes2 = nodes(tree2, t-1);
        for i in nodes1
            for j in nodes2
                # strange index in the children function root is implicitly numbered 0, conflict with function root() wich returns 1
                children1 = tree1.children[i+1];
                children2 = tree2.children[j+1];
                #p1 = tree1.probability[children1]
                #p2 = tree2.probability[children2]
                #s1 = children1
                #s2 = children2
                D[i,j] = C[i,j] = (Wasserstein(tree1.probability[children1], tree2.probability[children2], C[children1, children2].^r, 1.))
            end
        end
    end
    return (D[1,1])
end


function sinkhorn_nested_distance(tree1::Tree, tree2::Tree, max_nb_children, r, epsilon = 0 )
    """
        Computes the r-th Nested Distance between two given trees of same height for parameter r = 1 (default); 
    
        If epsilon == 0 (default value), then
        the regularization parameter λ of the Sinkhorn's algorithm
        is adapted to the distance matrix: λ = param_lamda*100/max(distance_ij) to avoid numerical instabilities.
    
        Else if epsilon != 0, then
        compute an epsilon/T-optimal solution for each regularized OT problem, which can be done by setting
        λ = epsilon/(4Tlog(n)) for enough iterations of Sinkhorn's algorithm. Exploits the log-sum-exp
        trick to handle small values of λ.
    """
    if epsilon == 0 # set λ = max(distance_ij) / 30 just to avoid numerical instabilities 
        T = height(tree1)+1; # T stage (the root is at height 0)
        D = zeros(Float64, (length(nodes(tree1)), length(nodes(tree2)))); # Alot will remain 0
        # Initialization, computes distance at time T 
        D[1,1] = (abs(tree1.state[1] - tree2.state[1]))^2;
        for t in 2:T
            nodes1 = nodes(tree1, t-1); # all nodes at stage t of tree1
            nodes2 = nodes(tree2, t-1);
            for i in nodes1
               for j in nodes2
                    D[i,j] = D[tree1.parent[i], tree2.parent[j]] + (abs(tree1.state[i] - tree2.state[j]))^2;
                end
            end
        end
        D.^=(1/2);
        # Recursive, backward in time
        for t in T-1:-1:1 
            nodes1 = nodes(tree1, t-1); 
            nodes2 = nodes(tree2, t-1);
            for i in nodes1
                for j in nodes2
                    # strange index in the children function root is implicitly numbered 0, conflict with function root() wich returns 1
                    children1 = tree1.children[i+1];
                    children2 = tree2.children[j+1];
                    λ =  maximum(D[children1, children2]) / 10;
                    D[i,j] = (Sinkhorn(tree1.probability[children1], 
                        tree2.probability[children2], distFunction(children1, children2),r, λ))^(1/r)   
                end
            end
        end
        return(D[1,1])
        else #WORK IN PROGRESS
        # epsilon > 0 and compute λ s.t. each entropic_OT returns an epsilon/T-optimal value,
        # exploits log-sum-exp trick to handle small values of λ.
        T = height(tree1)+1; # T stage (the root is at height 0)
        λ = epsilon/(4*T*log(max_nb_children));
        D = zeros(Float64, (length(nodes(tree1)), length(nodes(tree2)))); 
        D[1,1] = (abs(tree1.state[1] - tree2.state[1]))^2;
        for t in 2:T
            nodes1 = nodes(tree1, t-1); 
            nodes2 = nodes(tree2, t-1);
            for i in nodes1
               for j in nodes2
                    D[i,j] = D[tree1.parent[i], tree2.parent[j]] + (abs(tree1.state[i] - tree2.state[j]))^2;
                end
            end
        end
        D.^=(1/2);
        for t in T-1:-1:1 
            nodes1 = nodes(tree1, t-1); 
            nodes2 = nodes(tree2, t-1);
            for i in nodes1
                for j in nodes2
                    children1 = tree1.children[i+1];
                    children2 = tree2.children[j+1];
                    D[i,j] = (Sinkhorn(tree1.probability[children1], 
                    tree2.probability[children2], distFunction(children1, children2),r, -λ))^(1/r)   
                end
            end
        end
        return(D[1,1])
    end
end
