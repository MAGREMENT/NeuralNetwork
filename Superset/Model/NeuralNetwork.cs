using System.Runtime.InteropServices;

namespace Model;

public unsafe partial class NeuralNetwork : IDisposable
{
    private readonly void* _ptr;
    private readonly NeuralNetworkParameters _params = NeuralNetworkParameters.Default;

    public int Length => GetCount(_ptr);

    public NeuralNetwork(int[] layers)
    {
        _ptr = Initialize(layers.Length, layers);
    }

    public void ChangeParameters(Action<NeuralNetworkParameters> action)
    {
        action(_params);
    }

    public int GetNodeCount(int layer)
    {
        //TODO
        return 0;
    }

    public void SetWeight(int layer, int input, int output, int value)
    {
        //TODO
    }

    public int GetWeight(int layer, int input, int output)
    {
        //TODO
        return 0;
    }

    public double[] Predict(double[] inputs)
    {
        if (inputs.Length != GetNodeCount(0))
            throw new ArgumentException("Inputs length does not correspond to the first layer of the neural network");

        //TODO
        return Array.Empty<double>();
    }
    
    public void Dispose()
    {
        Dispose(_ptr);
    }

    [LibraryImport("libExport.dll")]
    [DefaultDllImportSearchPaths(DllImportSearchPath.AssemblyDirectory | DllImportSearchPath.ApplicationDirectory)]
    private static partial void* Initialize(int count, int[] layers);

    [LibraryImport("libExport.dll")]
    [DefaultDllImportSearchPaths(DllImportSearchPath.AssemblyDirectory | DllImportSearchPath.ApplicationDirectory)]
    private static partial void Dispose(void* ptr);

    [LibraryImport("libExport.dll")]
    [DefaultDllImportSearchPaths(DllImportSearchPath.AssemblyDirectory | DllImportSearchPath.ApplicationDirectory)]
    private static partial void ApplyParams(void* ptr, NeuralNetworkParameters parameters);

    [LibraryImport("libExport.dll")]
    [DefaultDllImportSearchPaths(DllImportSearchPath.AssemblyDirectory | DllImportSearchPath.ApplicationDirectory)]
    private static partial int GetCount(void* ptr);
}

[StructLayout(LayoutKind.Sequential)]
public struct NeuralNetworkParameters
{
    public double InitialLearningRate;
    public double LearningRateDecay;
    public double Regularization;
    public double Momentum;
    public int ActivationType;
    public int CostType;

    public static NeuralNetworkParameters Default { get; } = new()
    {
        InitialLearningRate = 1,
        LearningRateDecay = 0.002,
        Regularization = 0.9,
        Momentum = 0.1,
        ActivationType = Model.ActivationType.SIGMOID,
        CostType = Model.CostType.MEAN_SQUARED
    };
}

public static class ActivationType
{
    public const int DEFAULT = 0;
    public const int SIGMOID = 0;
    public const int TANH = 0;
    public const int RELU = 0;
    public const int SILU = 0;
    public const int SOFTMAX = 0;
}

public static class CostType
{
    public const int MEAN_SQUARED = 0;
}