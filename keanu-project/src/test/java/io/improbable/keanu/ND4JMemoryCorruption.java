package io.improbable.keanu;

public class ND4JMemoryCorruption {
    public static void main(String[] args) {
        String[] filenames = new String[]{
            "libwinpthread-1.dll",
            "libgcc_s_seh-1.dll",
            "libgomp-1.dll",
            "libstdc++-6.dll",
            "msvcr120.dll",
            "libiomp5md.dll",
            "mklml.dll",
            "libmkldnn.dll"
        };
        String path = "C:\\Users\\Charlie Crisp\\.javacpp\\cache\\mkl-dnn-0.16-1.4.3-windows-x86_64.jar\\org\\bytedeco\\javacpp\\windows-x86_64\\";
        for (String filename : filenames) {
            System.out.println("Loading " + filename);
            System.load(path + filename);
        }
        double smallValue = 1e-312;
        if (smallValue == 0) {
            throw new RuntimeException("Failure is before we load the ND4J lib");
        }

        System.load("C:\\Users\\Charlie Crisp\\.javacpp\\cache\\nd4j-native-1.0.0-beta3-windows-x86_64.jar\\org\\nd4j\\nativeblas\\windows-x86_64\\libnd4jcpu.dll");
        if (smallValue == 0) {
            throw new RuntimeException("Small numbers evaluate to zero!");
        }
    }
}
