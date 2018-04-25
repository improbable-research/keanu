package io.improbable.keanu.kotlin


class ArithmeticDouble(val value: Double) : DoubleOperators<ArithmeticDouble> {
    override fun exp(): ArithmeticDouble {
        return ArithmeticDouble(Math.exp(this.value))
    }

    override fun pow(that: ArithmeticDouble): ArithmeticDouble {
        return ArithmeticDouble(Math.pow(this.value, that.value))
    }

    override fun pow(value: Double): ArithmeticDouble {
        return ArithmeticDouble(Math.pow(this.value, value))
    }

    override fun log(): ArithmeticDouble {
        return ArithmeticDouble(Math.log(this.value))
    }

    override fun sin(): ArithmeticDouble {
        return ArithmeticDouble(Math.sin(this.value))
    }

    override fun cos(): ArithmeticDouble {
        return ArithmeticDouble(Math.cos(this.value))
    }

    override fun asin(): ArithmeticDouble {
        return ArithmeticDouble(Math.asin(this.value))
    }

    override fun acos(): ArithmeticDouble {
        return ArithmeticDouble(Math.acos(this.value))
    }

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
