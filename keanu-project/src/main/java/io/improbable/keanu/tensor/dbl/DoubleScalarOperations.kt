package io.improbable.keanu.tensor.dbl

import io.improbable.keanu.tensor.NumberScalarOperations
import org.apache.commons.math3.util.FastMath

object DoubleScalarOperations : NumberScalarOperations<Double> {

    // Number Ops
    override fun sub(left: Double, right: Double): Double {
        return left - right
    }

    override fun add(left: Double, right: Double): Double {
        return left + right
    }

    override fun rsub(left: Double, right: Double): Double {
        return right - left
    }

    override fun mul(left: Double, right: Double): Double {
        return left * right
    }

    override fun div(left: Double, right: Double): Double {
        return left / right
    }

    override fun rdiv(left: Double, right: Double): Double {
        return right / left
    }

    override fun pow(left: Double, right: Double): Double {
        return FastMath.pow(left, right)
    }

    //Comparisons

    override fun equalToMask(left: Double, right: Double): Double {
        return if (left == right) 1.0 else 0.0
    }

    override fun gtMask(left: Double, right: Double): Double {
        return if (left > right) 1.0 else 0.0
    }

    override fun gteMask(left: Double, right: Double): Double {
        return if (left >= right) 1.0 else 0.0
    }

    override fun ltMask(left: Double, right: Double): Double {
        return if (left < right) 1.0 else 0.0
    }

    override fun lteMask(left: Double, right: Double): Double {
        return if (left <= right) 1.0 else 0.0
    }

    override fun equalTo(left: Double, right: Double): Boolean {
        return left == right
    }

    override fun gt(left: Double, right: Double): Boolean {
        return left > right
    }

    override fun gte(left: Double, right: Double): Boolean {
        return left >= right
    }

    override fun lt(left: Double, right: Double): Boolean {
        return left < right
    }

    override fun lte(left: Double, right: Double): Boolean {
        return left <= right
    }

}