package io.improbable.keanu.kotlin

import io.improbable.keanu.DeterministicRule
import io.improbable.keanu.vertices.dbl.DoubleVertex
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex
import io.improbable.keanu.vertices.intgr.IntegerVertex
import io.improbable.keanu.vertices.intgr.nonprobabilistic.ConstantIntegerVertex
import junit.framework.TestCase
import org.junit.Rule
import org.junit.Test

class SimpleKotlinModelTest {

    @Rule
    @JvmField
    var deterministicRule = DeterministicRule()

    @Test
    fun runSimpleModel() {
        val modelA = SimpleModel<ArithmeticDouble, ArithmeticInteger>()

        val ad1 = ArithmeticDouble(1.0)
        val ad2 = ArithmeticDouble(1.5)
        val resultD1 = modelA.add(ad1, ad2)

        val ai1 = ArithmeticInteger(1)
        val ai2 = ArithmeticInteger(2)
        val resultI1 = modelA.add(ai1, ai2)

        val modelB = SimpleModel<DoubleVertex, IntegerVertex>()

        val dv1 = ConstantDoubleVertex(1.0)
        val dv2 = ConstantDoubleVertex(1.5)
        val resultD2 = modelB.add(dv1, dv2)

        val iv1 = ConstantIntegerVertex(1)
        val iv2 = ConstantIntegerVertex(2)
        val resultI2 = modelB.add(iv1, iv2)

        TestCase.assertEquals(resultD1.value, resultD2.value.scalar())
        TestCase.assertEquals(resultI1.value, resultI2.value.scalar())
    }

    class SimpleModel<A : DoubleOperators<A>, B : IntegerOperators<B>> {

        fun add(a: A, b: A): A {
            return a + b
        }

        fun add(a: B, b: B): B {
            return a + b
        }
    }
}
