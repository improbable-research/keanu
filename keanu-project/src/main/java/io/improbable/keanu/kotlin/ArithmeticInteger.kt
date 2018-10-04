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

    override fun minus(that: ArithmeticInteger) = ArithmeticInteger(value - that.value)

    override fun plus(that: ArithmeticInteger) = ArithmeticInteger(value + that.value)

    override fun times(that: ArithmeticInteger) = ArithmeticInteger(value * that.value)

    override fun div(that: ArithmeticInteger) = ArithmeticInteger(value / that.value)

    override fun minus(value: Int) = ArithmeticInteger(this.value - value)

    override fun intMinusThis(that: Int) = ArithmeticInteger(that - this.value)

    override fun plus(value: Int) = ArithmeticInteger(this.value + value)

    override fun times(value: Int) = ArithmeticInteger(this.value * value)

    override fun div(value: Int) = ArithmeticInteger(this.value / value)

    override fun intDivThis(that: Int) = ArithmeticInteger(that / this.value)

    override fun unaryMinus() = ArithmeticInteger(value) * -1

    override fun pow(exponent: ArithmeticInteger) = ArithmeticInteger(IntMath.pow(this.value, exponent.value))

    override fun pow(exponent: Int) = ArithmeticInteger(IntMath.pow(this.value, exponent))

}
