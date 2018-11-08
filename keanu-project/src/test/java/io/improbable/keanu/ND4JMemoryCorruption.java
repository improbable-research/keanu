package io.improbable.keanu;

public class ND4JMemoryCorruption {
    public static void main(String[] args) {
        String filename = "C:\\Users\\Charlie Crisp\\.javacpp\\cache\\nd4j-native-1.0.0-beta2-windows-x86_64.jar\\org\\nd4j\\nativeblas\\windows-x86_64\\libnd4jcpu.dll";
        System.load(filename);
        double smallValue = 1e-312;
        if (smallValue == 0) {
            throw new RuntimeException("Small numbers evaluate to zero!");
        }
    }
}
