﻿"""
    strip_arg(arg::Symbol)
Strip anything extra (type annotations, default values, etc) from an argument.
For now this cannot handle keyword arguments (it will throw an error).
"""
strip_arg(arg::Symbol) = arg # once we have a symbol, we have stripped everything, so we can just return it
function strip_arg(arg_expr::Expr)
    if arg_expr.head == :parameters # keyword argument
        error("strip_arg can't handle keyword args yet (parsing arg expression $(arg_expr))")
    elseif arg_expr.head == :(::) # argument is type annotated, remove the annotation
        return strip_arg(arg_expr.args[1])
    elseif arg_expr.head == :kw # argument has a default value, remove the default
        return strip_arg(arg_expr.args[1])
    else
        error("strip_arg encountered something unexpected. arg_expr was $(arg_expr)")
    end
end

"""
    required_func(signature)
Provide a default function implementation that throws an error when called.
"""
macro required_func(signature)
    if signature.head == :(=) # in this case a default implementation has already been supplied
        return esc(signature)
    end
    @assert signature.head == :call

    # get the names of all the arguments
    args = [strip_arg(expr) for expr in signature.args[2:end]]

    # get the name of the function
    fname = signature.args[1]

    error_string = "BayesNets.jl: No implementation of $fname for "

    # add each of the arguments to error string
    for (i,a) in enumerate(args)
        error_string *= "$a::\$(typeof($a))"
        if i == length(args)-1
            error_string *= ", and "
        elseif i != length(args)
            error_string *= ", "
        else
            error_string *= "."
        end
    end

    # if you are modifying this and want to debug, it might be helpful to print
    # println(error_string)

    body = Expr(:call, :error, parse("\"$error_string\""))

    return Expr(:function, esc(signature), esc(body))
end