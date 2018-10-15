package io.improbable.keanu.kotlin


interface NumberOperators<T> {

    operator fun minus(that: T): T

    operator fun plus(that: T): T

    operator fun times(that: T): T

    operator fun div(that: T): T

    operator fun unaryMinus(): T

    fun pow(exponent: T): T

}
