package io.improbable.keanu.randomfactory

import io.improbable.keanu.kotlin.ArithmeticDouble
import io.improbable.keanu.kotlin.DoubleOperators
import io.improbable.keanu.vertices.dbl.DoubleVertex
import io.improbable.keanu.vertices.dbl.KeanuRandom
import junit.framework.TestCase
import org.junit.Test

class RandomFactoryTest {

    @Test
    fun arithmeticAndVertexModelsAreEquivalent() {
        val modelA = getArithmeticDoubleTestModel()
        val modelB = getDoubleTensorVertexTestModel()

        val resultA: ArithmeticDouble = modelA.addGaussians(1.0, 1.0, 1.0)
        val resultB: DoubleVertex = modelB.addGaussians(1.0, 1.0, 1.0)

        TestCase.assertEquals(resultA.value, resultB.value.scalar(), 0.0)
    }

    @Test
    fun testDefault() {
        val modelA = getArithmeticDoubleTestModel()
        val modelB = getDoubleTensorVertexTestModel()

        var sumA = 0.0
        var sumB = 0.0
        val num = 10000

        for (i: Int in 0..num) {
            sumA += modelA.addDefaultGaussians().value
            sumB += modelB.addDefaultGaussians().value.scalar()
        }

        val meanA = sumA / num
        val meanB = sumB / num

        TestCase.assertEquals(meanA, meanB, 0.0)
    }

    private fun getArithmeticDoubleTestModel(): SimpleModel<ArithmeticDouble> {
        val arithmeticDoubleFactory = RandomDoubleFactory()
        arithmeticDoubleFactory.setRandom(KeanuRandom(1))

        return SimpleModel<ArithmeticDouble>(arithmeticDoubleFactory)
    }

    private fun getDoubleTensorVertexTestModel(): SimpleModel<DoubleVertex> {
        val doubleVertexFactory = DoubleVertexFactory()
        doubleVertexFactory.setRandom(KeanuRandom(1))

        return SimpleModel<DoubleVertex>(doubleVertexFactory)
    }

    inner class SimpleModel<T : DoubleOperators<T>>(val randomFactory: RandomFactory<T>) {

        fun addGaussians(muA: Double, muB: Double, sigma: Double): T {
            val a = randomFactory.nextGaussian(muA, sigma)
            val b = randomFactory.nextGaussian(muB, sigma)
            return a + b
        }

        fun addDefaultGaussians(): T {
            val a = randomFactory.nextGaussian()
            val b = randomFactory.nextGaussian()
            return a + b
        }
    }
}
