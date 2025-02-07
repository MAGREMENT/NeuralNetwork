using System.Runtime.InteropServices;

namespace Model;

public partial class NeuralNetwork : IDisposable
{
    private readonly IntPtr _ptr;
    private readonly NeuralNetworkParameters _params;

    public int Length => GetCount(_ptr);

    public NeuralNetwork(params int[] layers)
    {
        _ptr = Initialize(layers.Length, layers);
        _params = NeuralNetworkParameters.Default;
        ApplyParams(_ptr, _params);
    }

    private NeuralNetwork(IntPtr ptr, NeuralNetworkParameters parameters)
    {
        _ptr = ptr;
        _params = parameters;
        ApplyParams(_ptr, parameters);
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

    public void Randomize(double min, double max)
    {
        Randomize(_ptr, min, max);
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

    public void SetWeight(int layer, int input, int output, double value)
    {
        CheckBounds(layer, input, output);
        SetWeight(_ptr, layer, input, output, value);
    }

    public double GetWeight(int layer, int input, int output)
    {
        CheckBounds(layer, input, output);
        return GetWeight(_ptr, layer, input, output);
    }
    
    public void SetBias(int layer, int output, double value)
    {
        if (output < 0 || output >= GetOutCount(layer)) throw new IndexOutOfRangeException();
        SetBias(_ptr, layer, output, value);
    }
    
    public double GetBias(int layer, int output)
    {
        if (output < 0 || output >= GetOutCount(layer)) throw new IndexOutOfRangeException();
        return GetBias(_ptr, layer, output);
    }

    public double[] Predict(double[] inputs)
    {
        if (inputs.Length != GetInCount(0))
            throw new ArgumentException("Inputs length does not correspond to the first layer of the neural network");
        
        var count = GetOutCount(Length - 1);
        var arr = new double[count];
        
        Predict(_ptr, inputs, inputs.Length, arr, count);
        return arr;
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
    private static partial IntPtr Initialize(int count, int[] layers);

    [LibraryImport("libExport.dll")]
    [DefaultDllImportSearchPaths(DllImportSearchPath.AssemblyDirectory | DllImportSearchPath.ApplicationDirectory)]
    private static partial void Dispose(IntPtr ptr);

    [LibraryImport("libExport.dll")]
    [DefaultDllImportSearchPaths(DllImportSearchPath.AssemblyDirectory | DllImportSearchPath.ApplicationDirectory)]
    private static partial void ApplyParams(IntPtr ptr, NeuralNetworkParameters parameters);

    [LibraryImport("libExport.dll")]
    [DefaultDllImportSearchPaths(DllImportSearchPath.AssemblyDirectory | DllImportSearchPath.ApplicationDirectory)]
    private static partial int GetCount(IntPtr ptr);
    
    [LibraryImport("libExport.dll")]
    [DefaultDllImportSearchPaths(DllImportSearchPath.AssemblyDirectory | DllImportSearchPath.ApplicationDirectory)]
    private static partial int GetOutCount(IntPtr ptr, int layer);
    
    [LibraryImport("libExport.dll")]
    [DefaultDllImportSearchPaths(DllImportSearchPath.AssemblyDirectory | DllImportSearchPath.ApplicationDirectory)]
    private static partial int GetInCount(IntPtr ptr, int layer);
    
    [LibraryImport("libExport.dll")]
    [DefaultDllImportSearchPaths(DllImportSearchPath.AssemblyDirectory | DllImportSearchPath.ApplicationDirectory)]
    private static partial void SetWeight(IntPtr ptr, int layer, int input, int output, double value);
    
    [LibraryImport("libExport.dll")]
    [DefaultDllImportSearchPaths(DllImportSearchPath.AssemblyDirectory | DllImportSearchPath.ApplicationDirectory)]
    private static partial double GetWeight(IntPtr ptr, int layer, int input, int output);
    
    [LibraryImport("libExport.dll")]
    [DefaultDllImportSearchPaths(DllImportSearchPath.AssemblyDirectory | DllImportSearchPath.ApplicationDirectory)]
    private static partial void SetBias(IntPtr ptr, int layer, int output, double value);
    
    [LibraryImport("libExport.dll")]
    [DefaultDllImportSearchPaths(DllImportSearchPath.AssemblyDirectory | DllImportSearchPath.ApplicationDirectory)]
    private static partial double GetBias(IntPtr ptr, int layer, int output);
    
    [LibraryImport("libExport.dll")]
    [DefaultDllImportSearchPaths(DllImportSearchPath.AssemblyDirectory | DllImportSearchPath.ApplicationDirectory)]
    private static partial void Predict(IntPtr ptr, 
        [In, Out, MarshalAs(UnmanagedType.LPArray, SizeParamIndex = 2)] double[] input, int inCount,
        [In, Out, MarshalAs(UnmanagedType.LPArray, SizeParamIndex = 4)] double[] output, int outCount);

    [LibraryImport("libExport.dll", StringMarshalling = StringMarshalling.Utf8)]
    [DefaultDllImportSearchPaths(DllImportSearchPath.AssemblyDirectory | DllImportSearchPath.ApplicationDirectory)]
    private static partial IntPtr FromFile(string file, ref NeuralNetworkParameters parameters);
    
    [LibraryImport("libExport.dll", StringMarshalling = StringMarshalling.Utf8)]
    [DefaultDllImportSearchPaths(DllImportSearchPath.AssemblyDirectory | DllImportSearchPath.ApplicationDirectory)]
    private static partial void Save(IntPtr ptr, NeuralNetworkParameters parameters, string file);
    
    [LibraryImport("libExport.dll")]
    [DefaultDllImportSearchPaths(DllImportSearchPath.AssemblyDirectory | DllImportSearchPath.ApplicationDirectory)]
    private static partial void Randomize(IntPtr ptr, double min, double max);
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
    public const int SIGMOID = 1;
    public const int TANH = 2;
    public const int RELU = 3;
    public const int SILU = 4;
    public const int SOFTMAX = 5;
}

public static class CostType
{
    public const int MEAN_SQUARED = 0;
}