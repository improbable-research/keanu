package io.improbable.keanu.kotlin

import io.improbable.keanu.vertices.intgr.IntegerVertex
import io.improbable.keanu.vertices.intgr.nonprobabilistic.ConstantIntegerVertex


operator fun Int.plus(that: IntegerVertex): IntegerVertex {
    return that + this
}

operator fun Int.minus(that: IntegerVertex): IntegerVertex {
    return ConstantIntegerVertex(this) - that
}

operator fun Int.times(that: IntegerVertex): IntegerVertex {
    return that * this
}

operator fun Int.div(that: IntegerVertex): IntegerVertex {
    return ConstantIntegerVertex(this) / that
}

operator fun Int.plus(that: ArithmeticInteger): ArithmeticInteger {
    return that + this
}

operator fun Int.minus(that: ArithmeticInteger): ArithmeticInteger {
    return ArithmeticInteger(this) - that
}

operator fun Int.times(that: ArithmeticInteger): ArithmeticInteger {
    return that * this
}

operator fun Int.div(that: ArithmeticInteger): ArithmeticInteger {
    return ArithmeticInteger(this) / that
}
