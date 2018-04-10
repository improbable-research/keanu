package io.improbable.keanu.kotlin


interface IntegerOperators<T> : Operators<T> {

    operator fun minus(value: Int): T
    operator fun plus(value: Int): T
    operator fun times(value: Int): T
    operator fun div(value: Int): T

}
