package io.improbable.keanu.kotlin


interface DoubleOperators<T> : NumberOperators<T> {

    operator fun plus(that: Double): T

    operator fun minus(that: Double): T

    /**
     * @return that - this
     */
    fun reverseMinus(that: Double): T

    operator fun times(that: Double): T

    operator fun div(that: Double): T

    /**
     * @return that / this
     */
    fun reverseDiv(that: Double): T

    fun pow(exponent: Double): T

    fun log(): T

    fun sin(): T

    fun cos(): T

    fun asin(): T

    fun acos(): T

    fun exp(): T

}

operator fun <T : DoubleOperators<T>> Double.plus(that: T) = that + this

operator fun <T : DoubleOperators<T>> Double.minus(that: T) = that.reverseMinus(this)

operator fun <T : DoubleOperators<T>> Double.times(that: T) = that * this

operator fun <T : DoubleOperators<T>> Double.div(that: T) = that.reverseDiv(this)

fun <T : DoubleOperators<T>> pow(base: T, exponent: T) = base.pow(exponent)

fun <T : DoubleOperators<T>> pow(base: T, exponent: Double) = base.pow(exponent)

fun <T : DoubleOperators<T>> log(that: T) = that.log()

fun <T : DoubleOperators<T>> exp(that: T) = that.exp()

fun <T : DoubleOperators<T>> sin(that: T) = that.sin()

fun <T : DoubleOperators<T>> cos(that: T) = that.cos()

fun <T : DoubleOperators<T>> asin(that: T) = that.asin()

fun <T : DoubleOperators<T>> acos(that: T) = that.acos()
