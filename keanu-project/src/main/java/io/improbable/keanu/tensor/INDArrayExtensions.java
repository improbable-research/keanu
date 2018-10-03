package io.improbable.keanu.tensor;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.ops.transforms.Transforms;

public class INDArrayExtensions {

  public static INDArray castToInteger(INDArray tensor, boolean duplicate) {
    INDArray tensorToDropFractionOn = duplicate ? tensor.dup() : tensor;
    INDArray sign = Transforms.sign(tensorToDropFractionOn);
    Transforms.floor(Transforms.abs(tensorToDropFractionOn, false), false).muli(sign);
    return tensorToDropFractionOn;
  }
}
