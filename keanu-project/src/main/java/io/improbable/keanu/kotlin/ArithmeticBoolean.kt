package io.improbable.keanu.kotlin

/**
 * A simple implementation of [BooleanOperators] backed by a single [Boolean].
 *
 * This allows you to use vertices and plain old numbers in the same, generic code. e.g.
 *
 *    fun <T> isTrue(to: BooleanOperators<T>) = to.not()
 *
 *    // Both valid
 *    isTrue(ConstantBoolVertex(false))
 *    isTrue(ArithmeticBoolean(false))
 */
data class ArithmeticBoolean(val value: Boolean) : BooleanOperators<ArithmeticBoolean> {

    override fun and(that: ArithmeticBoolean) = ArithmeticBoolean(this.value && that.value)

    override fun and(that: Boolean) = ArithmeticBoolean(this.value && that)

    override fun or(that: ArithmeticBoolean) = ArithmeticBoolean(this.value || that.value)

    override fun or(that: Boolean) = ArithmeticBoolean(this.value || value)

    override fun not() = ArithmeticBoolean(!this.value)

}
