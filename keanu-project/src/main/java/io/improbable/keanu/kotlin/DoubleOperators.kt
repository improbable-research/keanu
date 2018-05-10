package io.improbable.keanu.kotlin


interface DoubleOperators<T> : Operators<T> {

    operator fun minus(that: Double): T
    operator fun plus(that: Double): T
    operator fun times(that: Double): T
    operator fun div(that: Double): T
    fun pow(exponent: T): T
    fun pow(exponent: Double): T
    fun log(): T
    fun sin(): T
    fun cos(): T
    fun asin(): T
    fun acos(): T
    fun exp(): T
}
