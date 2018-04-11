package io.improbable.keanu.kotlin


class ArithmeticDouble(val value: Double) : DoubleOperators<ArithmeticDouble> {

    override fun minus(that: ArithmeticDouble): ArithmeticDouble {
        return ArithmeticDouble(value - that.value)
    }

    override fun plus(that: ArithmeticDouble): ArithmeticDouble {
        return ArithmeticDouble(value + that.value)
    }

    override fun times(that: ArithmeticDouble): ArithmeticDouble {
        return ArithmeticDouble(value * that.value)
    }

    override fun div(that: ArithmeticDouble): ArithmeticDouble {
        return ArithmeticDouble(value / that.value)
    }

    override fun minus(value: Double): ArithmeticDouble {
        return ArithmeticDouble(this.value - value)
    }

    override fun plus(value: Double): ArithmeticDouble {
        return ArithmeticDouble(this.value + value)
    }

    override fun times(value: Double): ArithmeticDouble {
        return ArithmeticDouble(this.value * value)
    }

    override fun div(value: Double): ArithmeticDouble {
        return ArithmeticDouble(this.value / value)
    }

    override fun unaryMinus(): ArithmeticDouble {
        return this * -1.0
    }

}
