package io.improbable.keanu.kotlin


interface IntegerOperators<T> : NumberOperators<T> {

    operator fun plus(value: Int): T

    operator fun minus(value: Int): T

    /**
     * @return that - this
     */
    fun reverseMinus(that: Int): T

    operator fun times(value: Int): T

    operator fun div(value: Int): T

    /**
     * @return that / this
     */
    fun reverseDiv(that: Int): T

    fun pow(exponent: Int): T

}

operator fun <T : IntegerOperators<T>> Int.plus(that: T) = that + this

operator fun <T : IntegerOperators<T>> Int.minus(that: T) = that.reverseMinus(this)

operator fun <T : IntegerOperators<T>> Int.times(that: T) = that * this

operator fun <T : IntegerOperators<T>> Int.div(that: T) = that.reverseDiv(this)
