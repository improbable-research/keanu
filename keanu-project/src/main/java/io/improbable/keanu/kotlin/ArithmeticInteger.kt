package io.improbable.keanu.kotlin

import com.google.common.math.IntMath

/**
 * A simple implementation of [IntegerOperators] backed by a single [Int].
 *
 * This allows you to use vertices and plain old numbers in the same, generic code. e.g.
 *
 *    fun <T> add5(to: IntegerOperators<T>) = to + 5
 *
 *    // Both valid
 *    add5(ConstantIntegerVertex(0))
 *    add5(ArithmeticInteger(5))
 */
data class ArithmeticInteger(val value: Int) : IntegerOperators<ArithmeticInteger> {

    override fun minus(that: ArithmeticInteger) = this.minus(that.value)

    override fun plus(that: ArithmeticInteger) = this.plus(that.value)

    override fun times(that: ArithmeticInteger) = this.times(that.value)

    override fun div(that: ArithmeticInteger) = this.div(that.value)

    override fun minus(value: Int) = ArithmeticInteger(this.value - value)

    override fun reverseMinus(that: Int) = ArithmeticInteger(that - this.value)

    override fun plus(value: Int) = ArithmeticInteger(this.value + value)

    override fun times(value: Int) = ArithmeticInteger(this.value * value)

    override fun div(value: Int) = ArithmeticInteger(this.value / value)

    override fun reverseDiv(that: Int) = ArithmeticInteger(that / this.value)

    override fun unaryMinus() = ArithmeticInteger(-this.value)

    override fun pow(exponent: ArithmeticInteger) = this.pow(exponent.value);

    override fun pow(exponent: Int) = ArithmeticInteger(IntMath.pow(this.value, exponent))

}
