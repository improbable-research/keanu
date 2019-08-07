package io.improbable.keanu.tensor

interface FloatingPointScalarOperations<T : Number> : NumberScalarOperations<T> {

    fun sigmoid(value: T): T

    fun reciprocal(value: T): T

    fun sqrt(value: T): T

    fun log(value: T): T

    fun logGamma(value: T): T

    fun digamma(value: T): T

    fun trigamma(value: T): T

    fun sin(value: T): T

    fun cos(value: T): T

    fun tan(value: T): T

    fun atan(value: T): T

    fun atan2(x: T, y: T): T

    fun asin(value: T): T

    fun acos(value: T): T

    fun sinh(value: T): T

    fun cosh(value: T): T

    fun tanh(value: T): T

    fun asinh(value: T): T

    fun acosh(value: T): T

    fun atanh(value: T): T

    fun exp(value: T): T

    fun log1p(value: T): T

    fun log2(value: T): T

    fun log10(value: T): T

    fun exp2(value: T): T

    fun expM1(value: T): T

    fun ceil(value: T): T

    fun floor(value: T): T

    fun round(value: T): T

    fun safeLogTimes(left: T, right: T): T

    fun logAddExp(left: T, right: T): T

    fun logAddExp2(left: T, right: T): T

    fun notNan(value: T): Boolean

    fun isNaN(value: T): Boolean

    fun isFinite(value: T): Boolean

    fun isInfinite(value: T): Boolean

    fun isNegativeInfinity(value: T): Boolean

    fun isPositiveInfinity(value: T): Boolean

}