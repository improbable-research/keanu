package io.improbable.keanu.kotlin

fun <T : DoubleOperators<T>> pow(base: T, exponent: T): T {
    return base.pow(exponent)
}

fun <T : DoubleOperators<T>> pow(base: T, exponent: Double): T {
    return base.pow(exponent)
}

fun <T : DoubleOperators<T>> log(that: T): T {
    return that.log()
}

fun <T : DoubleOperators<T>> exp(that: T): T {
    return that.exp()
}

fun <T : DoubleOperators<T>> sin(that: T): T {
    return that.sin()
}

fun <T : DoubleOperators<T>> cos(that: T): T {
    return that.cos()
}

fun <T : DoubleOperators<T>> asin(that: T): T {
    return that.asin()
}

fun <T : DoubleOperators<T>> acos(that: T): T {
    return that.acos()
}

