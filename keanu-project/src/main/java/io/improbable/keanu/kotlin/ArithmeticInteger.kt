package io.improbable.keanu.kotlin


class ArithmeticInteger(val value: Int) : IntegerOperators<ArithmeticInteger> {

    override fun minus(that: ArithmeticInteger): ArithmeticInteger {
        return ArithmeticInteger(value - that.value)
    }

    override fun plus(that: ArithmeticInteger): ArithmeticInteger {
        return ArithmeticInteger(value + that.value)
    }

    override fun times(that: ArithmeticInteger): ArithmeticInteger {
        return ArithmeticInteger(value * that.value)
    }

    override fun div(that: ArithmeticInteger): ArithmeticInteger {
        return ArithmeticInteger(value / that.value)
    }

    override fun minus(value: Int): ArithmeticInteger {
        return ArithmeticInteger(this.value - value)
    }

    override fun plus(value: Int): ArithmeticInteger {
        return ArithmeticInteger(this.value + value)
    }

    override fun times(value: Int): ArithmeticInteger {
        return ArithmeticInteger(this.value * value)
    }

    override fun div(value: Int): ArithmeticInteger {
        return ArithmeticInteger(this.value / value)
    }

    override fun unaryMinus(): ArithmeticInteger {
        return ArithmeticInteger(value) * -1
    }

}
