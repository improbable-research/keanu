package io.improbable.keanu.kotlin


interface DoubleOperators<T> : Operators<T> {

    operator fun minus(value: Double): T
    operator fun plus(value: Double): T
    operator fun times(value: Double): T
    operator fun div(value: Double): T

}
