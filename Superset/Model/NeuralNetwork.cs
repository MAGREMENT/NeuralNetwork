using System.Runtime.InteropServices;

namespace Model;

public unsafe partial class NeuralNetwork : IDisposable
{
    private readonly void* _ptr;
    private readonly NeuralNetworkParameters _params;

    public int Length => GetCount(_ptr);

    public NeuralNetwork(int[] layers)
    {
        _ptr = Initialize(layers.Length, layers);
        _params = NeuralNetworkParameters.Default;
    }

    private NeuralNetwork(void* ptr, NeuralNetworkParameters parameters)
    {
        _ptr = ptr;
        _params = parameters;
    }

    public static NeuralNetwork Import(string file)
    {
        var p = new NeuralNetworkParameters();
        var ptr = FromFile(file, ref p);
        return new NeuralNetwork(ptr, p);
    }

    public void Save(string file)
    {
        Save(_ptr, _params, file);
    }

    public void ChangeParameters(Action<NeuralNetworkParameters> action)
    {
        action(_params);
        ApplyParams(_ptr, _params);
    }

    public int GetInCount(int layer)
    {
        if (layer < 0 || layer >= GetCount(_ptr)) throw new IndexOutOfRangeException();
        return GetInCount(_ptr, layer);
    }

    public int GetOutCount(int layer)
    {
        if (layer < 0 || layer >= GetCount(_ptr)) throw new IndexOutOfRangeException();
        return GetOutCount(_ptr, layer);
    }

    public void SetWeight(int layer, int input, int output, int value)
    {
        CheckBounds(layer, input, output);
        SetWeight(_ptr, layer, input, output, value);
    }

    public int GetWeight(int layer, int input, int output)
    {
        CheckBounds(layer, input, output);
        return GetWeight(_ptr, layer, input, output);
    }
    
    public void SetBias(int layer, int output, int value)
    {
        if (output < 0 || output >= GetOutCount(layer)) throw new IndexOutOfRangeException();
        SetBias(_ptr, layer, output, value);
    }
    
    public int GetBias(int layer, int output)
    {
        if (output < 0 || output >= GetOutCount(layer)) throw new IndexOutOfRangeException();
        return GetBias(_ptr, layer, output);
    }

    public double[] Predict(double[] inputs)
    {
        if (inputs.Length != GetInCount(0))
            throw new ArgumentException("Inputs length does not correspond to the first layer of the neural network");

        return Predict(_ptr, inputs, GetInCount(0), GetOutCount(Length - 1));
    }
    
    public void Dispose()
    {
        GC.SuppressFinalize(this);
        Dispose(_ptr);
    }

    private void CheckBounds(int layer, int input, int output)
    {
        if (input < 0 || input >= GetInCount(layer) || output < 0 || output >= GetOutCount(layer))
            throw new IndexOutOfRangeException();
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
    
    [LibraryImport("libExport.dll")]
    [DefaultDllImportSearchPaths(DllImportSearchPath.AssemblyDirectory | DllImportSearchPath.ApplicationDirectory)]
    private static partial int GetOutCount(void* ptr, int layer);
    
    [LibraryImport("libExport.dll")]
    [DefaultDllImportSearchPaths(DllImportSearchPath.AssemblyDirectory | DllImportSearchPath.ApplicationDirectory)]
    private static partial int GetInCount(void* ptr, int layer);
    
    [LibraryImport("libExport.dll")]
    [DefaultDllImportSearchPaths(DllImportSearchPath.AssemblyDirectory | DllImportSearchPath.ApplicationDirectory)]
    private static partial void SetWeight(void* ptr, int layer, int input, int output, int value);
    
    [LibraryImport("libExport.dll")]
    [DefaultDllImportSearchPaths(DllImportSearchPath.AssemblyDirectory | DllImportSearchPath.ApplicationDirectory)]
    private static partial int GetWeight(void* ptr, int layer, int input, int output);
    
    [LibraryImport("libExport.dll")]
    [DefaultDllImportSearchPaths(DllImportSearchPath.AssemblyDirectory | DllImportSearchPath.ApplicationDirectory)]
    private static partial void SetBias(void* ptr, int layer, int output, int value);
    
    [LibraryImport("libExport.dll")]
    [DefaultDllImportSearchPaths(DllImportSearchPath.AssemblyDirectory | DllImportSearchPath.ApplicationDirectory)]
    private static partial int GetBias(void* ptr, int layer, int output);
    
    [LibraryImport("libExport.dll")]
    [return : MarshalAs(UnmanagedType.LPArray, SizeParamIndex = 3)]
    [DefaultDllImportSearchPaths(DllImportSearchPath.AssemblyDirectory | DllImportSearchPath.ApplicationDirectory)]
    private static partial  double[] Predict(void* ptr, double[] input, int inCount, int outCount);

    [LibraryImport("libExport.dll", StringMarshalling = StringMarshalling.Utf8)]
    [DefaultDllImportSearchPaths(DllImportSearchPath.AssemblyDirectory | DllImportSearchPath.ApplicationDirectory)]
    private static partial void* FromFile(string file, ref NeuralNetworkParameters parameters);
    
    [LibraryImport("libExport.dll", StringMarshalling = StringMarshalling.Utf8)]
    [DefaultDllImportSearchPaths(DllImportSearchPath.AssemblyDirectory | DllImportSearchPath.ApplicationDirectory)]
    private static partial void Save(void* ptr, NeuralNetworkParameters parameters, string file);
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