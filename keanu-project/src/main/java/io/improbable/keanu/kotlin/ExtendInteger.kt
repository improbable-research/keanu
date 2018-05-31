package io.improbable.keanu.kotlin

import io.improbable.keanu.tensor.intgr.IntegerTensor
import io.improbable.keanu.vertices.intgr.IntegerVertex
import io.improbable.keanu.vertices.intgr.nonprobabilistic.ConstantIntegerVertex


//TODO: remove when removing nontensor integers
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

// Tensor versions
operator fun Int.plus(that: io.improbable.keanu.vertices.intgrtensor.IntegerVertex): io.improbable.keanu.vertices.intgrtensor.IntegerVertex {
    return that + this
}

operator fun Int.minus(that: io.improbable.keanu.vertices.intgrtensor.IntegerVertex): io.improbable.keanu.vertices.intgrtensor.IntegerVertex {
    return io.improbable.keanu.vertices.intgrtensor.nonprobabilistic.ConstantIntegerVertex(this) - that
}

operator fun Int.times(that: io.improbable.keanu.vertices.intgrtensor.IntegerVertex): io.improbable.keanu.vertices.intgrtensor.IntegerVertex {
    return that * this
}

operator fun Int.div(that: io.improbable.keanu.vertices.intgrtensor.IntegerVertex): io.improbable.keanu.vertices.intgrtensor.IntegerVertex {
    return io.improbable.keanu.vertices.intgrtensor.nonprobabilistic.ConstantIntegerVertex(this) / that
}
///

operator fun Int.plus(that: IntegerTensor): IntegerTensor {
    return that + this
}

operator fun Int.minus(that: IntegerTensor): IntegerTensor {
    return -that + this
}

operator fun Int.times(that: IntegerTensor): IntegerTensor {
    return that * this
}

operator fun Int.div(that: IntegerTensor): IntegerTensor {
    return IntegerTensor.scalar(this) / that
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
