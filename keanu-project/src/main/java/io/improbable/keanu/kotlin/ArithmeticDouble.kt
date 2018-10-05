package io.improbable.keanu.kotlin

/**
 * A simple implementation of [DoubleOperators] backed by a single [Double].
 *
 * This allows you to use vertices and plain old numbers in the same, generic code. e.g.
 *
 *    fun <T> add5(to: DoubleOperators<T>) = to + 5.0
 *
 *    // Both valid
 *    add5(ConstantDoubleVertex(0.0))
 *    add5(ArithmeticDouble(5.0))
 */
data class ArithmeticDouble(val value: Double) : DoubleOperators<ArithmeticDouble> {

    override fun exp() = ArithmeticDouble(Math.exp(this.value))

    override fun pow(exponent: ArithmeticDouble) = this.pow(exponent.value)

    override fun pow(exponent: Double) = ArithmeticDouble(Math.pow(this.value, exponent))

    override fun log() = ArithmeticDouble(Math.log(this.value))

    override fun sin() = ArithmeticDouble(Math.sin(this.value))

    override fun cos() = ArithmeticDouble(Math.cos(this.value))

    override fun asin() = ArithmeticDouble(Math.asin(this.value))

    override fun acos() = ArithmeticDouble(Math.acos(this.value))

    override fun minus(that: ArithmeticDouble) = this.minus(that.value)

    override fun plus(that: ArithmeticDouble) = this.plus(that.value)

    override fun times(that: ArithmeticDouble) = this.times(that.value)

    override fun div(that: ArithmeticDouble) = this.div(that.value)

    override fun minus(that: Double) = ArithmeticDouble(this.value - that)

    override fun doubleMinusThis(that: Double) = ArithmeticDouble(that - this.value)

    override fun plus(that: Double) = ArithmeticDouble(this.value + that)

    override fun times(that: Double) = ArithmeticDouble(this.value * that)

    override fun doubleDivThis(that: Double) = ArithmeticDouble(that / this.value)

    override fun div(that: Double) = ArithmeticDouble(this.value / that)

    override fun unaryMinus() = ArithmeticDouble(-this.value)

}
