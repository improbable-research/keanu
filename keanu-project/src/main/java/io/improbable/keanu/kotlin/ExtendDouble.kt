package io.improbable.keanu.kotlin

import io.improbable.keanu.tensor.dbl.DoubleTensor
import io.improbable.keanu.vertices.dbltensor.DoubleTensorVertex
import io.improbable.keanu.vertices.dbltensor.nonprobabilistic.ConstantDoubleTensorVertex

// Vertices
operator fun Double.plus(that: DoubleTensorVertex): DoubleTensorVertex {
    return that + this
}

operator fun Double.minus(that: DoubleTensorVertex): DoubleTensorVertex {
    return ConstantDoubleTensorVertex(this) - that
}

operator fun Double.times(that: DoubleTensorVertex): DoubleTensorVertex {
    return that * this
}

operator fun Double.div(that: DoubleTensorVertex): DoubleTensorVertex {
    return ConstantDoubleTensorVertex(this) / that
}

// Tensors
operator fun Double.plus(that: DoubleTensor): DoubleTensor {
    return that + this
}

operator fun Double.minus(that: DoubleTensor): DoubleTensor {
    return -that + this
}

operator fun Double.times(that: DoubleTensor): DoubleTensor {
    return that * this
}

operator fun Double.div(that: DoubleTensor): DoubleTensor {
    return DoubleTensor.scalar(this) / that
}

// Other Arithmetic Doubles
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
