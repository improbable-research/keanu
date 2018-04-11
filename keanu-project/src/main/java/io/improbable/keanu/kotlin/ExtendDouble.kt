package io.improbable.keanu.kotlin

import io.improbable.keanu.vertices.dbl.DoubleVertex
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex


operator fun Double.plus(that: DoubleVertex): DoubleVertex {
    return that + this
}

operator fun Double.minus(that: DoubleVertex): DoubleVertex {
    return ConstantDoubleVertex(this) - that
}

operator fun Double.times(that: DoubleVertex): DoubleVertex {
    return that * this
}

operator fun Double.div(that: DoubleVertex): DoubleVertex {
    return ConstantDoubleVertex(this) / that
}

operator fun Double.plus(that: ArithmeticDouble): ArithmeticDouble {
    return that + this
}

operator fun Double.minus(that: ArithmeticDouble): ArithmeticDouble {
    return ArithmeticDouble(this) - that
}

operator fun Double.times(that: ArithmeticDouble): ArithmeticDouble {
    return ArithmeticDouble(this) - that
}

operator fun Double.div(that: ArithmeticDouble): ArithmeticDouble {
    return ArithmeticDouble(this) / that
}
