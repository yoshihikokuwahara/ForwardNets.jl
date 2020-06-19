abstract type Node{T<:Real} end

abstract type Layer{T} <: Node{T} end
abstract type Activation{T} <: Node{T} end

const NameOrIndex = Union{Symbol,Int}

Base.eltype{T}(::Node{T}) = T

"""
    name(n::Node)
Return the Symbol name of this Node.
"""
@required_func name(n::Node)

"""
    output(n::Node)
Get the tensor output of the Node
"""
@required_func output(n::Node)


"""
    forward!(n::Node)
Compute the node output
"""
@required_func forward!(n::Node)
